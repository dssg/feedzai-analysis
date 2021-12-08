import psycopg2
import yaml

def get_db_con(creds_file):
    """ Get an authenticated psycopg db connection, given a credentials file"""

    try: 
        with open (creds_file, 'r') as file:
            creds = yaml.safe_load(file)['db']
    except Exception as e:
        print(e)
        print('Error reading the config file')
    
    connection = psycopg2.connect(
        user=creds['user'],
        password=creds['pass'],
        host=creds['host'],
        port=creds['port'],
        database=creds['db']
    )

    return connection