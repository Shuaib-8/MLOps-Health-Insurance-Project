# src/features/engineer.py
import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("health-insurance-feature-engineering")


def create_preprocessor(multi_cat_strategy="ordinal"):
    """
    Create a preprocessing pipeline for health insurance data.

    Args:
        multi_cat_strategy: Strategy for multi-categorical encoding of `region`
            - 'ordinal' (default): OrdinalEncoder (creates 1 column with 0,1,2,3)
            - 'onehot': OneHotEncoder (creates 4 columns)
    """
    logger.info(
        f"Creating preprocessor pipeline with {multi_cat_strategy} encoding for multi-categorical features"
    )

    # Define feature groups for the health insurance dataset
    # Binary categorical features - will use OrdinalEncoder
    binary_categorical_features = ["sex", "smoker"]

    # Multi-categorical features - encoding strategy depends on parameter
    multi_categorical_features = ["region"]

    logger.info(f"Binary categorical features: {binary_categorical_features}")
    logger.info(f"Multi-categorical features: {multi_categorical_features}")

    # Numerical features (no transformation needed)
    numerical_features = ["age", "bmi", "children"]

    # Preprocessing for binary categorical features
    # Use OrdinalEncoder instead of LabelEncoder for pipeline compatibility
    binary_transformer = make_pipeline(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )

    # Preprocessing for multi-categorical features based on strategy
    if multi_cat_strategy == "ordinal":
        multi_categorical_transformer = make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
    elif multi_cat_strategy == "onehot":
        multi_categorical_transformer = make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )
    else:
        raise ValueError(f"Unknown multi_cat_strategy: {multi_cat_strategy}")

    # Combine preprocessors in a column transformer
    preprocessor = make_column_transformer(
        ("passthrough", numerical_features),
        (binary_transformer, binary_categorical_features),
        (multi_categorical_transformer, multi_categorical_features),
        remainder="drop",  # Drop any other columns
    )

    return preprocessor, multi_cat_strategy


def run_feature_engineering(
    input_file, output_file, preprocessor_file, encoding_strategy="ordinal"
):
    """Full feature engineering pipeline for health insurance data."""
    # Load cleaned data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Original dataset shape: {df.shape}")

    # Create and fit the preprocessor for all features
    preprocessor, strategy = create_preprocessor(multi_cat_strategy=encoding_strategy)

    # Separate features and target
    target_col = "charges"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None

    # Fit and transform using the preprocessor
    X_transformed = preprocessor.fit_transform(X)

    logger.info("Fitted the preprocessor and transformed all features")

    # Get feature names from the preprocessor
    feature_names = []

    # Add numerical feature names (passthrough)
    feature_names.extend(["age", "bmi", "children"])

    # Add binary encoded feature names (OrdinalEncoder output)
    feature_names.extend(["sex", "smoker"])

    # Add multi-categorical feature names based on strategy
    if strategy == "onehot":
        # Add one-hot encoded feature names (all 4 regions)
        if len(preprocessor.transformers_) > 2:
            multi_cat_transformer = preprocessor.transformers_[2][1]
            if hasattr(multi_cat_transformer, "steps"):
                onehot_encoder = multi_cat_transformer.steps[-1][1]
                multi_cat_features = ["region"]
                onehot_feature_names = onehot_encoder.get_feature_names_out(
                    multi_cat_features
                )
                feature_names.extend(onehot_feature_names)
    elif strategy == "ordinal":
        # Single encoded column for region
        feature_names.extend(["region"])

    logger.info(f"Final feature names: {feature_names}")

    # Create final dataframe
    df_final = pd.DataFrame(X_transformed, columns=feature_names)
    if y is not None:
        df_final[target_col] = y.values

    logger.info(f"Final dataset shape: {df_final.shape}")
    logger.info(f"Encoding strategy used: {strategy}")

    # Log all encoding mappings for easy reference
    logger.info("=" * 50)
    logger.info("ENCODING MAPPINGS SUMMARY")
    logger.info("=" * 50)

    # Always log binary encoder mappings first
    binary_transformer = preprocessor.transformers_[1][1]
    binary_encoder = binary_transformer.steps[-1][1]

    if hasattr(binary_encoder, "categories_"):
        sex_mapping = {
            cat: idx for idx, cat in enumerate(binary_encoder.categories_[0])
        }
        smoker_mapping = {
            cat: idx for idx, cat in enumerate(binary_encoder.categories_[1])
        }

        logger.info(f"Binary Categorical Mappings:")
        logger.info(f"  Sex: {sex_mapping}")
        logger.info(f"  Smoker: {smoker_mapping}")

    # Save the complete preprocessor with mappings
    preprocessor_data = {
        "preprocessor": preprocessor,
        "encoding_strategy": strategy,
        "feature_names": feature_names,
    }

    # Add and log encoding mappings for interpretability
    if strategy == "ordinal":
        # Get the ordinal encoder for multi-categorical features
        multi_cat_transformer = preprocessor.transformers_[2][1]
        ordinal_encoder = multi_cat_transformer.steps[-1][1]

        # Create mapping dictionary
        region_mapping = {}
        if hasattr(ordinal_encoder, "categories_"):
            categories = ordinal_encoder.categories_[0]  # First (and only) feature
            for idx, category in enumerate(categories):
                region_mapping[category] = idx

        preprocessor_data["region_mapping"] = region_mapping

        # Create reverse mapping for logging
        reverse_mapping = {v: k for k, v in region_mapping.items()}

        logger.info(f"Multi-Categorical Ordinal Mappings:")
        logger.info(f"  Region: {region_mapping}")
        logger.info(
            f"  Region Reverse (0=northeast, 1=northwest, etc.): {reverse_mapping}"
        )

    elif strategy == "onehot":
        # Get the one-hot encoder mappings
        multi_cat_transformer = preprocessor.transformers_[2][1]
        onehot_encoder = multi_cat_transformer.steps[-1][1]

        if hasattr(onehot_encoder, "categories_"):
            region_categories = onehot_encoder.categories_[0]
            preprocessor_data["region_categories"] = list(region_categories)

            logger.info(f"Multi-Categorical One-Hot Mappings:")
            logger.info(f"  Region categories: {list(region_categories)}")
            logger.info(
                f"  Created columns: {[name for name in feature_names if 'region_' in name]}"
            )

    # Add binary mappings to saved data
    if hasattr(binary_encoder, "categories_"):
        preprocessor_data["sex_mapping"] = sex_mapping
        preprocessor_data["smoker_mapping"] = smoker_mapping

    logger.info(f"Final feature names: {feature_names}")
    logger.info("=" * 50)

    joblib.dump(preprocessor_data, preprocessor_file)
    logger.info(f"Saved complete preprocessor with mappings to {preprocessor_file}")

    # Save fully preprocessed data
    df_final.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")

    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering for health insurance data."
    )
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file")
    parser.add_argument(
        "--output", required=True, help="Path for output CSV file (engineered features)"
    )
    parser.add_argument(
        "--preprocessor", required=True, help="Path for saving the preprocessor"
    )
    parser.add_argument(
        "--encoding",
        choices=["ordinal", "onehot"],
        default="ordinal",
        help="Encoding strategy for multi-categorical features (default: ordinal)",
    )

    args = parser.parse_args()

    run_feature_engineering(args.input, args.output, args.preprocessor, args.encoding)
