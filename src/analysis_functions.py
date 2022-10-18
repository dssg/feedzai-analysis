import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import itertools

from sklearn.utils import resample


schemas = [
    "ab_server_cmu_share_control", 
    "ab_server_cmu_share_explainers",
    "ab_server_cmu_share_random_all",
    "ab_server_cmu_share_random_j",
    "ab_server_cmu_share_random_n",
    "ab_server_cmu_share_random_t",
]

users = (
    "exp_ns_fa_j",
    "exp_ns_fa_n",
    "exp_ns_fa_t",
    "exp_sc_fa_j",
    "exp_sc_fa_n",
    "exp_sc_fa_t",
    "exp_3_fa_j",
    "exp_3_fa_n",
    "exp_3_fa_t",
    "exp_4_fa_j",
    "exp_4_fa_n",
    "exp_4_fa_t",
    "exp_5_fa_j",
    "exp_5_fa_n",
    "exp_5_fa_t",
    "exp_6_fa_j",
    "exp_6_fa_n",
    "exp_6_fa_t",
    "exp_7_fa_j",
    "exp_7_fa_n",
    "exp_7_fa_t",
    "exp_8_fa_j",
    "exp_8_fa_n",
    "exp_8_fa_t",
)

# Mappping the db name to a "nicer" name
# groups = {
#     'Control_fa_ns': 'Control-A',
#     'Control_fa_scores': 'Control-B',
#     'Lime_1': 'LIME',
#     'FZRF_TreeInterpreter_1': 'TreeInt',
#     'TreeShap_1': 'TreeSHAP',
#     'random_explainer_1': 'Random'
# }

# Mapping to names for the paper
groups = {
    'Control_fa_ns': 'Data',
    'Control_fa_scores': 'ML Model',
    'Lime_1': 'LIME',
    'FZRF_TreeInterpreter_1': 'TreeInt',
    'TreeShap_1': 'TreeSHAP',
    'random_explainer_1': 'Random'
}



# group_order = [
#     'Control-A', 'Control-B', 'TreeSHAP', 'TreeInt', 'LIME', 'Random', 'Irrelevant'
# ]

group_order = [
    'Data', 'ML Model', 'TreeSHAP', 'LIME', 'TreeInt', 'Random', 'Irrelevant'
]

colors = [
    sns.color_palette("colorblind")[0],
    sns.color_palette("colorblind")[9],
    sns.color_palette("colorblind")[4],
    sns.color_palette("colorblind")[2],
    sns.color_palette("colorblind")[3],
    sns.color_palette("colorblind")[8],
    sns.color_palette("colorblind")[7],

]


def _get_variant_from_random_explainer(db_conn, row):

    random_explainer_schemas = [
        "ab_server_cmu_share_random_all",
        "ab_server_cmu_share_random_j",
        "ab_server_cmu_share_random_n",
        "ab_server_cmu_share_random_t"
    ]
    if row["group"] != "random_explainer_1":
        return row["group"]
    
    for schema in random_explainer_schemas:
        openxai_schema = "openxai" + schema[9:]
        query = f"""
            SELECT 
                ec.explanation_value ->> 'if_random' as random 
            FROM {schema}.trx_user_assignments as tua
            LEFT JOIN {schema}.explanation_assignments as ea using (xplz_id)
            LEFT JOIN {openxai_schema}.explainers as e
                on (ea.explanation ->> 'explainer_id')::integer = e.explainer_id
            LEFT JOIN {openxai_schema}.explanation_components as ec
                on ec.explainer_uuid = e.explainer_uuid and
                tua.trx_id = ec.trx_id
            WHERE tua.xplz_id = '{row["xplz_id"]}'
        """
        values = pd.read_sql(query, db_conn)["random"].values.tolist()
        if values:
            if values[0] == "false":
                return 'Random'
            elif values[0] == "true":
                return 'Irrelevant'
    
    return row["group"]


def _remove_duplicates(group_df):
    """The way we are fetching the records from different schema results in duplicates. Removing them"""
    d = dict()
    group_df['trx_amnt'].fillna(0, inplace=True)
    group_df['label'].fillna(0, inplace=True)
    d['trx_amnt'] = group_df['trx_amnt'].max()
    d['decision'] = group_df['decision'].max()
    d['decision_time'] = group_df['decision_time'].max()
    d['label'] = group_df['label'].max()

    return pd.Series(d)
    

def get_all_decisions(conn, schemas, users, groups):
    """Fetch all the decisions from the DB as a dataframe

        return: A dataframe with the following
            - id of the trx (xplz_id)
            - user identifier (the suffix, either n, j, or t)
            - group
            - trx value
            - decision
            - decision_time
            - label
    """
    qs = [
        f"""
        select distinct on (rf.xplz_id)
            rf.xplz_id, 
            rf.user_name,
            rf."group",
            ua.label_fields -> 'amount_usd' as trx_amnt,
            rf.decision,
            tf.duration as decision_time,
            ua.label
        from {schema}.review_feedback rf
        left join {schema}.trx_user_assignments ua using(xplz_id, user_id)
        left join {schema}.time_feedback tf using(xplz_id, user_id)
        where tf.duration IS NOT NULL
        AND rf.user_name in {users} 
        AND rf."group" in {tuple(groups.keys())}
        ORDER BY rf.xplz_id, rf.submission_datetime DESC

        """
        for schema in schemas
    ]

    # print(qs)

    df_all = pd.concat([pd.read_sql(q, conn) for q in qs])
    df_all['user_name'] = df_all['user_name'].str.split('_').apply(lambda x: '_'.join(x[-1:]))

    df_all['group'] = df_all.apply(lambda x: _get_variant_from_random_explainer(conn, x), axis=1 )

    # adding the nicer name
    df_all['group'] = df_all['group'].apply(lambda x: groups[x] if x in groups else x)

    df_all = df_all.groupby(['xplz_id', 'group', 'user_name']).apply(_remove_duplicates).reset_index()

    return df_all


def get_all_explanations(conn, schemas, groups):
    """fetch all the explanations for each explainer"""


def assign_conf_mat_cell(decisions, suspicious_strategy='correct' ):
    if suspicious_strategy=='correct':
        def process_decision(row):
            label = row["label"]
            decision = row["decision"]
            
            if label == 0 and decision in {"approved", "suspicious"}:
                return "tn"
            elif label == 1 and decision == "approved":
                return "fn"
            elif label == 0 and decision == "declined":
                return "fp"
            elif label == 1 and decision in {"declined", "suspicious"}:
                return "tp"
        
                
    elif suspicious_strategy=='approve':
        def process_decision(row):
            label = row["label"]
            decision = row["decision"]
            
            if label == 0 and decision in {"approved", "suspicious"}:
                return "tn"
            elif label == 1 and decision in {"approved", "suspicious"}:
                return "fn"
            elif label == 0 and decision == "declined":
                return "fp"
            elif label == 1 and decision == "declined":
                return "tp"
    
    elif suspicious_strategy=='decline':
        def process_decision(row):
            label = row["label"]
            decision = row["decision"]
            
            if label == 0 and decision == "approved":
                return "tn"
            elif label == 1 and decision == "approved":
                return "fn"
            elif label == 0 and decision in {"declined", "suspicious"}:
                return "fp"
            elif label == 1 and decision in {"declined", "suspicious"}:
                return "tp"
    else:
        raise ValueError('Invalid strategy to handle suspicious')

    conf_mat = decisions.apply(lambda x: process_decision(x), axis=1)

    return conf_mat

    
def _modify_value_and_time(decisions, params, suspicious_strategy):
    """
    Calcualte the value of the decision and apply any time penalities
    based on where the decision falls in the conf mat
    """

    dc = decisions.copy()
    
    dc['conf_mat'] = assign_conf_mat_cell(dc, suspicious_strategy)

    coeffs = {
        'tp': 0,
        'tn': 1 + (params['cust_worth']) * params['p_return_cust'],
        'fn': params['fn'],
        'fp': (1 - params['p_loss_trx']) + (params['cust_worth'] * (1-params['p_loss_cust'])) # assuming the customer worth is on average n times the transaction value
    }

    # calculating the decision value
    dc['decision_value'] = dc.apply(lambda x: coeffs[x['conf_mat']]*x['trx_amnt'], axis=1)

    # We apply the penaly only if we treat the suspicious ones as a 
    if suspicious_strategy=='correct':
        # penalizing the time for suspicious
        dc['modified_time'] = dc.apply(lambda x: x['decision_time'] + params['suspicious_add_time']  if x['decision']=='suspicious' else x['decision_time'], axis=1)
    else:
        dc['modified_time'] = dc['decision_time']

        
    return dc


def dps(decisions, params, suspicious_strategy, agg_levels=['group'], use_cov=False):
    """
        Calculate the dollar per second metric at the given aggregate level
    """

    dcs = _modify_value_and_time(decisions, params, suspicious_strategy)

    grpobj = dcs.groupby(agg_levels)

    dps = list()

    for g, df in grpobj:
        mean_d = df['decision_value'].mean()
        mean_t = df['modified_time'].mean()

        var_d = df['decision_value'].std() ** 2
        var_t = df['modified_time'].std() ** 2

        # calculating the metric
        metric = mean_d / mean_t

        if use_cov:
            t = np.cov(df['decision_value'], df['modified_time']) 
            cov = abs(t[0, 1])
        else:
            cov = 0

        # variance of the ratio
        var = (1 / mean_t**2) * var_d + (mean_d**2 / mean_t**4) * var_t - 2 * (mean_d / mean_t ** 3) * cov

        # standard error of the ratio
        se = np.sqrt(var/len(df))

        d = dict()

        if len(agg_levels) == 1:
            d[agg_levels[0]] = g

        else: 
            for i, agg in enumerate(agg_levels):
                d[agg] = g[i]
            
        d['mean'] = metric
        d['var'] = var
        d['se'] = se
        d['n'] = len(df)

        dps.append(d)

    return pd.DataFrame(dps)


def dt(decisions, params, suspicious_strategy, agg_levels=['group']):
    """
        decision time at the given aggregate level     
    """
    dcs = _modify_value_and_time(decisions, params, suspicious_strategy)

    grpobj = dcs.groupby(agg_levels)

    dt = list()
    for g, df in grpobj:
        d = dict()

        if len(agg_levels) == 1:
                    d[agg_levels[0]] = g
        else: 
            for i, agg in enumerate(agg_levels):
                d[agg] = g[i] 
        
        d['mean'] = df['modified_time'].mean()
        d['var'] = df['modified_time'].var()
        d['n'] = len(df)
        d['se'] = stats.sem(df['modified_time'])

        dt.append(d)

    return pd.DataFrame(dt)


def pdr(decisions, params, suspicious_strategy, agg_levels=['group'], n_samples=None, n_iterations=1000):
    """
    Calculating the percent dollar regret metric. 
    This calculates how far the decisions in the group are from a perfect scenario    
    Here, calculating the variance for the metric using bootstrapping 
    """

    dcs = _modify_value_and_time(decisions, params, suspicious_strategy)

    dcs['potential_revenue'] = dcs.apply(
        lambda x: x['trx_amnt'] * (1 + (params['cust_worth'] * params['p_return_cust'])) if x['label']==0 else 0, 
        axis=1
    )

    grpobj = dcs.groupby(agg_levels)

    pdr = list()
    for g, df in grpobj:
        means = list()

        if n_samples is None:
            n=len(df)
            print('n_samples not set. Defaulting to the group size')
        else:
            n=n_samples

        for i in range(n_iterations):
            t = resample(df, replace=True, n_samples=n)

            m = 1 - (t['decision_value'].sum() / t['potential_revenue'].sum())
            means.append(m)

        d = dict()
        if len(agg_levels) == 1:
            d[agg_levels[0]] = g
        else: 
            for i, agg in enumerate(agg_levels):
                d[agg] = g[i]

        d['mean'] = np.mean(means)
        d['n'] = len(df)
        d['se'] = np.std(means)
        d['var'] = len(df) * (np.std(means) ** 2)

        pdr.append(d)

    return pd.DataFrame(pdr)


def ttests_operational_metrics(metrics, comparisons):
    
    results = list()

    for comp in comparisons:
        mean1 = metrics[metrics['group'] == comp[0]]['mean'].iloc[0]
        mean2 = metrics[metrics['group'] == comp[1]]['mean'].iloc[0]
        
        var1 = metrics[metrics['group'] == comp[0]]['var'].iloc[0]
        var2 = metrics[metrics['group'] == comp[1]]['var'].iloc[0]

        n1 = metrics[metrics['group'] == comp[0]]['n'].iloc[0]
        n2 = metrics[metrics['group'] == comp[1]]['n'].iloc[0]
        
        res = stats.ttest_ind_from_stats(
            mean1, np.sqrt(var1), n1,
            mean2, np.sqrt(var2), n2,
        )
        
        
        d = dict()
        d['comparison'] = comp
        d['t-stat'] = res.statistic
        d['p-val'] = res.pvalue
        
        results.append(d)
        
    return pd.DataFrame(results)


def do_parameter_sweep(
    decisions, 
    comparisons,
    fn,
    p_loss_trx, 
    cust_worth,
    p_loss_cust,
    p_return_cust,
    suspicious_handling
):
    all_param_combinations = itertools.product(
        fn, 
        p_loss_trx, 
        cust_worth, 
        p_loss_cust, 
        p_return_cust,
        suspicious_handling
    )

    results = list()
    significance_results = list()
    for config in all_param_combinations:
        
        # suspicious handling
        if isinstance(config[5], int):
            add_time= config[5]
            suspicious_strategy = 'correct'
        else:
            add_time= 0 # The value doesn't matter
            suspicious_strategy = config[5]
            
        
        param = {
            'fn': config[0],
            'p_loss_trx': config[1],
            'cust_worth': config[2],
            'p_loss_cust': config[3],
            'p_return_cust': config[4],
            'suspicious_add_time': add_time,
            'suspicious_strategy': suspicious_strategy
        }

        metrics = list()
        ttests = list()
        
        df = dt(decisions, param, suspicious_strategy, ['group'])
        df['metric'] = 'dt'
        
        ttest = ttests_operational_metrics(df, comparisons)
        ttest['metric'] = 'dt'
        
        metrics.append(df)
        ttests.append(ttest)
        
        
        df = dps(decisions, param, suspicious_strategy, ['group'])
        df['metric'] = 'dps'
        
        ttest = ttests_operational_metrics(df, comparisons)
        ttest['metric'] = 'dps'
        
        metrics.append(df)
        ttests.append(ttest)
        
        df = pdr(decisions, param, suspicious_strategy, ['group'], n_samples=500, n_iterations=100)
        df['metric'] = 'pdr'
        
        ttest = ttests_operational_metrics(df, comparisons)
        ttest['metric'] = 'pdr'
        
        metrics.append(df)
        ttests.append(ttest)
        
        res = pd.concat(metrics)
        sig_res = pd.concat(ttests)
        
        # appending the parameter values
        for p, v in param.items():
            res[p] = v
            sig_res[p] = v


        results.append(res)
        significance_results.append(sig_res)

    
    return pd.concat(results), pd.concat(significance_results)


def chi_square_test(contingency_table: np.array) -> tuple:
    """ Performs a chi-squared test, given an matrix with the observed values
    (contingency table).

    It is assumed that the rows represent the group and the columns represent
    the classes.

    Parameters
    ----------
    contingency: numpy.array
        Observed frequencies of each group for a given class.

    Return
    ------
    tuple
        Statistic and p-value of the test.
    """
    expected = []
    for j in range(contingency_table.shape[0]):
        expected_line = []
        for i in range(contingency_table.shape[1]):
            frequency = np.sum(contingency_table[:, i]) / np.sum(contingency_table)
            expected_value = frequency * np.sum(contingency_table[j, :])
            expected_line.append(expected_value)
        expected.append(expected_line)
    expected = np.array(expected)
    statistic = np.sum((contingency_table-expected)**2 / expected)
    dof = (contingency_table.shape[0]-1) * (contingency_table.shape[1]-1)
    pvalue = 1 - stats.chi2(df=dof).cdf(statistic)
    return statistic, pvalue