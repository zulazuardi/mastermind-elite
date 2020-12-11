import os
import logging
import pickle
import numpy as np
import pandas as pd
import warnings

from logger import setup_logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from argparse import ArgumentParser

setup_logging(
    filename="classify"
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def get_year_month_mapping(filepath):
    """
    Get mapping table for year-month to integer
    Args:
        filepath  (str): The order dataset contains order_date column

    Returns:
        `year_month_mapping` (`dataframe`).
        the dataframe contain:
            - year_month : year month date format
            - x : integer unique mapping for each year_month
    """
    order_data = pd.read_csv(filepath)
    order_data["order_date"] = pd.to_datetime(
        order_data['order_date'], format="%Y-%m-%d")
    order_data["year_month"] = order_data[
        "order_date"].dt.to_period('M')
    year_month = pd.DataFrame({"year_month": pd.date_range(
        order_data["year_month"].min().to_timestamp(),
        order_data["year_month"].max().to_timestamp(), freq="MS",
        closed=None)})
    year_month["x"] = year_month.index + 1
    year_month["year_month"] = year_month["year_month"].dt.to_period('M')

    return year_month


def get_intercept(x, y):
    """
        Get intercept from 1D regression
        Args:
            x  (list): List of integer unique mapping for year_month
            y  (list): List of agg monthly attributes (can be frequency or
                       monetary value

        Returns:
            intercept  (float)
    """
    model = np.polyfit(x, y, 1)

    return float(model[1])


def get_slope(x, y):
    """
            Get slope from 1D regression
            Args:
                x  (list): List of integer unique mapping for year_month
                y  (list): List of agg monthly attributes (can be frequency or
                           monetary value

            Returns:
                slope  (float)
    """
    model = np.polyfit(x, y, 1)

    return float(model[0])


def get_label_dataset(filepath):
    """
        Get label dataset
        Args:
            filepath  (str): The label dataset filepath

        Returns:
            `label_data` (`dataframe`).
    """
    label_data = pd.read_csv(filepath)

    return label_data


def get_aggregate_order_dataset(filepath, is_failed=True):
    """
        Get agg order dataset by customer id
        Args:
            filepath  (str): The order dataset contains order_date column
            is_failed  (bool): boolean indicate whether the order was success
            or failed

        Returns:
            `final_data` (`dataframe`).
    """
    order_data = pd.read_csv(filepath)
    if is_failed:
        index_is_failed = 1
        prefix_label = "failed"
        logger.debug(
            "Generate failed order dataset"
        )
    else:
        index_is_failed = 0
        prefix_label = "success"
        logger.debug(
            "Generate success order dataset"
        )

    order_data = order_data[order_data["is_failed"] == index_is_failed]
    order_data["order_date"] = pd.to_datetime(
        order_data['order_date'], format="%Y-%m-%d")
    order_data["order_date_timestamp"] \
        = (
                  order_data['order_date'] - pd.Timestamp("1970-01-01")
          ) // pd.Timedelta('1s')
    order_data["diff_order_date"] = \
        order_data.groupby("customer_id")["order_date"].diff()
    order_data["diff_order_date"] = order_data["diff_order_date"].dt.days
    order_data["is_free_delivery"] = np.where(
        order_data['delivery_fee'] == 0,
        1, 0
    )
    order_data["is_promo"] = np.where(
        order_data['voucher_amount'] != 0,
        1, 0
    )
    order_data["is_breakfast_order"] = np.where(
        order_data['order_hour'].between(4, 11, inclusive=False),
        1, 0
    )
    order_data["is_lunch_order"] = np.where(
        order_data['order_hour'].between(10, 17, inclusive=False),
        1, 0
    )
    order_data["is_dinner_order"] = np.where(
        order_data['order_hour'].between(16, 23, inclusive=False),
        1, 0
    )
    order_data["is_supper_order"] = np.where(
        order_data['order_hour'].between(22, 5, inclusive=False),
        1, 0
    )
    order_data["year_month"] = order_data[
        "order_date"].dt.to_period('M')

    year_month = get_year_month_mapping(filepath)

    order_data = order_data.merge(year_month, on="year_month", how="left")

    agg_data = order_data.groupby("customer_id").agg({
        "order_date": ["count", "min", "max"],
        "order_date_timestamp": ["mean", "std", "median"],
        "order_hour": ["mean", "std", "median"],
        "diff_order_date": ["mean"],
        "is_free_delivery": "sum",
        "is_promo": "sum",
        "is_breakfast_order": "sum",
        "is_lunch_order": "sum",
        "is_dinner_order": "sum",
        "is_supper_order": "sum",
        "voucher_amount": ["mean", "sum"],
        "delivery_fee": ["mean", "sum"],
        "amount_paid": ["mean", "sum"],
        "restaurant_id": [pd.Series.nunique,
                          lambda x: x.value_counts().index[0]],
        "payment_id": [pd.Series.nunique, lambda x: x.value_counts().index[0]],
        "city_id": [pd.Series.nunique, lambda x: x.value_counts().index[0]],
        "platform_id": [pd.Series.nunique,
                        lambda x: x.value_counts().index[0]]
    })
    agg_data.columns = ["_".join(x) for x in agg_data.columns.ravel()]
    agg_data = agg_data.reset_index()

    reg_data = order_data.groupby(["customer_id", "x"]).agg({
        "order_date": "count",
        "voucher_amount": "sum",
        "delivery_fee": "sum",
        "amount_paid": "sum",
    })
    reg_data.columns = [x for x in reg_data.columns.ravel()]
    reg_data = reg_data.reset_index()
    reg_data = reg_data.groupby("customer_id").agg({
        "x": lambda x: list(x),
        "order_date": lambda x: list(x),
        "voucher_amount": lambda x: list(x),
        "delivery_fee": lambda x: list(x),
        "amount_paid": lambda x: list(x)
    })
    reg_data = reg_data.reset_index()
    reg_data["slope_order"] = np.vectorize(get_slope)(
        reg_data['x'], reg_data['order_date'])
    reg_data["slope_voucher_amount"] = np.vectorize(get_slope)(
        reg_data['x'], reg_data['voucher_amount'])
    reg_data["slope_delivery_fee"] = np.vectorize(get_slope)(
        reg_data['x'], reg_data['delivery_fee'])
    reg_data["slope_amount_paid"] = np.vectorize(get_slope)(
        reg_data['x'], reg_data['amount_paid'])
    reg_data["intercept_order"] = np.vectorize(get_intercept)(
        reg_data['x'], reg_data['order_date'])
    reg_data["intercept_voucher_amount"] = np.vectorize(get_intercept)(
        reg_data['x'], reg_data['voucher_amount'])
    reg_data["intercept_delivery_fee"] = np.vectorize(get_intercept)(
        reg_data['x'], reg_data['delivery_fee'])
    reg_data["intercept_amount_paid"] = np.vectorize(get_intercept)(
        reg_data['x'], reg_data['amount_paid'])
    reg_data = reg_data[[
        "customer_id",
        "slope_order",
        "slope_voucher_amount",
        "slope_delivery_fee",
        "slope_amount_paid",
        "intercept_order",
        "intercept_voucher_amount",
        "intercept_delivery_fee",
        "intercept_amount_paid"
    ]]
    reg_data = reg_data.rename(columns={
        "slope_order": "slope_{}_order".format(prefix_label),
        "slope_voucher_amount":
            "slope_voucher_amount_{}_order".format(prefix_label),
        "slope_delivery_fee":
            "slope_delivery_fee_{}_order".format(prefix_label),
        "slope_amount_paid":
            "slope_amount_paid_{}_order".format(prefix_label),
        "intercept_order": "intercept_order_{}_order".format(prefix_label),
        "intercept_voucher_amount":
            "intercept_voucher_amount_{}_order".format(prefix_label),
        "intercept_delivery_fee":
            "intercept_delivery_fee_{}_order".format(prefix_label),
        "intercept_amount_paid":
            "intercept_amount_paid_{}_order".format(prefix_label)
    })
    agg_data = agg_data.rename(columns={
        "order_date_count": "number_of_{}_order".format(prefix_label),
        "order_date_max": "last_{}_order".format(prefix_label),
        "order_date_min": "first_{}_order".format(prefix_label),
        "order_date_timestamp_mean":
            "avg_{}_order_timestamp".format(prefix_label),
        "order_date_timestamp_median":
            "median_{}_order_timestamp".format(prefix_label),
        "order_date_timestamp_std":
            "stddev_{}_order_timestamp".format(prefix_label),
        "order_hour_mean": "avg_{}_order_hour".format(prefix_label),
        "order_hour_median": "median_{}_order_hour".format(prefix_label),
        "order_hour_std": "stddev_{}_order_hour".format(prefix_label),
        "diff_order_date_mean":
            "avg_days_between_{}_order".format(prefix_label),
        "is_free_delivery_sum":
            "total_free_delivery_{}_order".format(prefix_label),
        "is_promo_sum": "total_promo_{}_order".format(prefix_label),
        "is_breakfast_order_sum":
            "total_breakfast_{}_order".format(prefix_label),
        "is_lunch_order_sum": "total_lunch_{}_order".format(prefix_label),
        "is_dinner_order_sum": "total_dinner_{}_order".format(prefix_label),
        "is_supper_order_sum": "total_supper_{}_order".format(prefix_label),
        "voucher_amount_mean":
            "avg_voucher_amount_{}_order".format(prefix_label),
        "voucher_amount_sum":
            "total_voucher_amount_{}_order".format(prefix_label),
        "delivery_fee_mean": "avg_delivery_fee_{}_order".format(prefix_label),
        "delivery_fee_sum": "total_delivery_fee_{}_order".format(prefix_label),
        "amount_paid_mean": "avg_amount_paid_{}_order".format(prefix_label),
        "amount_paid_sum": "total_amount_paid_{}_order".format(prefix_label),
        "restaurant_id_nunique":
            "number_distinct_restaurant_id_{}_order".format(prefix_label),
        "restaurant_id_<lambda_0>":
            "most_favourite_restaurant_id_{}_order".format(prefix_label),
        "payment_id_nunique":
            "number_distinct_payment_id_{}_order".format(prefix_label),
        "payment_id_<lambda_0>":
            "most_favourite_payment_id_{}_order".format(prefix_label),
        "city_id_nunique":
            "number_distinct_city_id_{}_order".format(prefix_label),
        "city_id_<lambda_0>":
            "most_favourite_city_id_{}_order".format(prefix_label),
        "platform_id_nunique":
            "number_distinct_platform_id_{}_order".format(prefix_label),
        "platform_id_<lambda_0>":
            "most_favourite_platform_id_{}_order".format(prefix_label),
    })
    agg_data = agg_data.fillna(0)
    reg_data = reg_data.fillna(0)

    final_data = agg_data.merge(reg_data, on="customer_id", how="left")
    logger.debug(
        "Finish generate dataset with number of row: {}".format(
            final_data["customer_id"].count()
        )
    )

    return final_data


def get_train_test_dataset(order_filename, label_filename, train_size=0.7):
    """
        Split dataset into training and testing datasets
        Args:
            order_filename  (str): The order dataset filename
            label_filename  (str): The label dataset filename
            train_size  (float): training size, default: 0.7

        Returns:
            `X_train`, `X_test`, `y_train`, `y_test` (`dataframe`).
    """
    success_order_data = get_aggregate_order_dataset(
        order_filename,
        is_failed=False
    )
    success_order_data["days_since_last_success_order"] = (
            pd.to_datetime(
                "2017-02-28",
                format="%Y-%m-%d"
            ) -
            success_order_data["last_success_order"]
    ).dt.days
    success_order_data["days_days_between_first_last_success_order"] = (
            success_order_data["last_success_order"] -
            success_order_data["first_success_order"]
    ).dt.days

    failed_order_data = get_aggregate_order_dataset(
        order_filename,
        is_failed=True
    )
    failed_order_data["days_since_last_failed_order"] = (
            pd.to_datetime(
                "2017-02-28",
                format="%Y-%m-%d"
            ) -
            failed_order_data["last_failed_order"]
    ).dt.days
    failed_order_data["days_days_between_first_last_failed_order"] = (
            failed_order_data["last_failed_order"] -
            failed_order_data["first_failed_order"]
    ).dt.days

    order_data = success_order_data.merge(
        failed_order_data,
        on="customer_id",
        how="outer"
    )

    order_data[[
        "days_since_last_success_order",
        "days_days_between_first_last_success_order",
        "days_since_last_failed_order",
        "days_days_between_first_last_failed_order"
    ]] = order_data[[
        "days_since_last_success_order",
        "days_days_between_first_last_success_order",
        "days_since_last_failed_order",
        "days_days_between_first_last_failed_order"
    ]].fillna(999999999)

    order_data = order_data.fillna(0)

    label_data = get_label_dataset(
        label_filename
    )

    dataset = order_data.merge(label_data, on="customer_id", how="left")
    X = dataset.drop([
        "customer_id",
        "first_success_order",
        "last_success_order",
        "first_failed_order",
        "last_failed_order",
        "is_returning_customer"
    ], axis=1)
    y = dataset[["is_returning_customer"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def grid_search_classify(train_features, test_features, train_labels,
                         test_labels, max_depths, n_estimators,
                         imbalance="oversampling"):
    """
        Training the model using grid search
        Args:
            train_features  (dataframe): Train features
            test_features  (dataframe): Test features
            train_labels  (dataframe): Train label
            test_labels  (dataframe): Test label
            max_depths  (dataframe): list of max depth for grid search
            n_estimators  (dataframe): list of n estimators for grid search
            imbalance  (str): Sampling method

        Returns:
            best model
    """
    param_grid = {
        'classification__max_depth': max_depths,
        'classification__n_estimators': n_estimators
    }
    if imbalance == "oversampling":
        steps = [('over', RandomOverSampler()),
                 ('classification', RandomForestClassifier())]
    else:
        steps = [('under', RandomUnderSampler()),
                 ('classification', RandomForestClassifier())]

    pipeline = Pipeline(steps=steps)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(train_features, train_labels)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(test_features)

    logger.debug(
        "classification report: {}".format(
            classification_report(
                test_labels["is_returning_customer"].values,
                predictions
            )
        )
    )

    return best_model


def save_model(model, model_name):
    """
        Save model to pickle file
        Args:
            model  : The model
            model_name  (str): The model name

        Returns:
            `X_train`, `X_test`, `y_train`, `y_test` (`dataframe`).
    """
    filename = os.path.join(
        "model/" + model_name + ".sav")
    pickle.dump(model, open(filename, 'wb'))


def main():
    parser = ArgumentParser()
    parser.add_argument("--imbalance", default='oversampling',
                        help="sampling method, "
                             "undersampling, oversampling")
    parser.add_argument("--model_name",
                        help="name of the model, the model will be saved in "
                             "/model folder")
    parser.add_argument("--max_depths", default='[50, 60, 70, 80]',
                        help="list of max depths, should be in string of list")
    parser.add_argument("--n_estimators", default='[300, 400, 500]',
                        help="list of n estimators, should be in string of "
                             "list")

    args = parser.parse_args()

    imbalance = args.imbalance
    model_name = args.model_name
    max_depths = args.max_depths
    n_estimators = args.n_estimators

    max_depths = max_depths.strip('][').split(', ')
    n_estimators = n_estimators.strip('][').split(', ')

    max_depths = [int(i) for i in max_depths]
    n_estimators = [int(i) for i in n_estimators]

    logger.info(
        "Start training the model with imbalance: {}, model_name: {}, "
        "max_depths: {}, and n_estimators: {}".format(
            imbalance, model_name, max_depths, n_estimators
        )
    )

    train_features, test_features, train_labels, test_labels = \
        get_train_test_dataset(
            "data/machine_learning_challenge_order_data.csv.gz",
            "data/machine_learning_challenge_labeled_data.csv.gz",
            0.7
        )

    model = grid_search_classify(
        train_features,
        test_features,
        train_labels,
        test_labels,
        max_depths=max_depths,
        n_estimators=n_estimators,
        imbalance=imbalance
    )

    save_model(model, model_name)

    logger.info(
        "Done training the model"
    )


if __name__ == "__main__":
    main()
