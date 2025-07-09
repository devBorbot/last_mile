import pandas as pd

def clean_data(df):
    df = df.dropna(
        subset=
        [
            'order_id',
            'pickup_time',
            'pickup_lng', 
            'pickup_lat',
            'delivery_time', 
            'delivery_lng',
            'delivery_lat'
        ]
    )
    
    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df.loc[:, 'duration_minutes'] = (df['delivery_time'] - df['pickup_time']).dt.total_seconds() / 60
    return df.reset_index(drop=True)
