import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from paths import CORE_DIR

class LOLWinProbabilityModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.models = {}
        self.timepoints = [10, 15, 20, 25]


    def load_and_filter_data(self):
        df = pd.read_csv(self.csv_path)
        df = df[(df['position'] == 'team') & (df['datacompleteness'] == 'complete')]
        return df


    def reduce_features(self, df):
        features = [
            'result', 'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs',
            'towers', 'opp_towers'
        ]

        for t in self.timepoints:
            features.extend([
                f'goldat{t}', f'xpat{t}', f'csat{t}', f'opp_goldat{t}', f'opp_xpat{t}', f'opp_csat{t}',
                f'golddiffat{t}', f'xpdiffat{t}', f'csdiffat{t}', f'killsat{t}', f'assistsat{t}', f'deathsat{t}',
                f'opp_killsat{t}', f'opp_assistsat{t}', f'opp_deathsat{t}'
            ])

        df_reduced = df[features].copy()

        return df_reduced
    

    def get_feature_columns(self, t):
        return [
            'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs',
            'towers', 'opp_towers',
            f'goldat{t}', f'xpat{t}', f'csat{t}', f'opp_goldat{t}', f'opp_xpat{t}', f'opp_csat{t}',
            f'golddiffat{t}', f'xpdiffat{t}', f'csdiffat{t}', f'killsat{t}',
            f'opp_killsat{t}'
        ]


    def train_models(self):
        df = self.load_and_filter_data()
        df_reduced = self.reduce_features(df)

        for t in self.timepoints:
            feature_cols = self.get_feature_columns(t)
            df_filtered = df_reduced[feature_cols + ['result']].dropna()

            X = df_filtered[feature_cols]
            y = df_filtered['result']

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(max_iter=1000))
            ])

            pipeline.fit(X, y)
            self.models[t] = pipeline
            joblib.dump(pipeline, f'logistic_model_{t}min.pkl')


    def load_models(self):
        for t in self.timepoints:
            self.models[t] = joblib.load(CORE_DIR/f'logistic_model_{t}min.pkl')


    def predict_win_probability(self, data_sample: pd.DataFrame, timepoint: int):
        if timepoint not in self.models:
            raise ValueError(f"Model for {timepoint} minutes not loaded.")

        model = self.models[timepoint]
        scaler = model.named_steps['scaler']

        data_sample_scaled = scaler.transform(data_sample)

        return model.predict_proba(data_sample_scaled)[:, 1]


    def train_models_with_performance(self):
        df = self.load_and_filter_data()
        df_reduced = self.reduce_features(df)
        df_reduced.dropna()
        print(len(df_reduced))
        for t in self.timepoints:
            feature_cols = self.get_feature_columns(t)

            df_filtered = df_reduced[feature_cols + ['result']].dropna()
            print(f"{t} length = {len(df_filtered)}")
            X = df_filtered[feature_cols]
            y = df_filtered['result']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(max_iter=1000))
            ])

            pipeline.fit(X_train, y_train)
            accuracy = pipeline.score(X_test, y_test)
            print(f"Model for {t} minutes trained with accuracy: {accuracy:.4f}")

            self.models[t] = pipeline
            joblib.dump(pipeline, f'logistic_model_{t}min.pkl')



def sample_to_dataframe(sample, t=10):
    """
    Converts the constructed sample dictionary into a DataFrame for model prediction.
    Missing features are filled with zeros if not provided.

    Args:
        sample (dict): Dictionary of extracted features from screenshots.
        minute (int): The closest time frame minute (10, 15, 20, or 25).

    Returns:
        pd.DataFrame: Single-row DataFrame in the expected format for prediction.
    """

    feature_cols = [
        'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs',
        'towers', 'opp_towers',
        f'goldat{t}', f'xpat{t}', f'csat{t}', f'opp_goldat{t}', f'opp_xpat{t}', f'opp_csat{t}',
        f'golddiffat{t}', f'xpdiffat{t}', f'csdiffat{t}', f'killsat{t}',
        f'opp_killsat{t}'
    ]

    filled_sample = {col: sample.get(col, 0) for col in feature_cols}

    df = pd.DataFrame([filled_sample])

    return df


if __name__ == "__main__":
    model_trainer = LOLWinProbabilityModel("../data/gametime_data.csv")
    model_trainer.train_models()
    #model_trainer.train_models_with_performance()
    model_trainer.load_models()

