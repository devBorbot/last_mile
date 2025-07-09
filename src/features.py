import numpy as np

def add_time_features(df):
    # Extract hour and weekday
    df['hour'] = df['pickup_time'].dt.hour
    df['weekday'] = df['pickup_time'].dt.weekday

    # Circular encoding for hour (24 hours in a day)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Circular encoding for weekday (7 days in a week)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    return df

def add_distance_feature(df):
    # Haversine distance between pickup and delivery
    def haversine(row):
        lat1, lon1, lat2, lon2 = map(
            np.radians, 
            [
                row['pickup_lat'],
                row['pickup_lng'],
                row['delivery_lat'],
                row['delivery_lng']
            ]
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # Earth radius in km
    df['distance_km'] = df.apply(haversine, axis=1)
    return df
