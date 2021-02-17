"""
pandera schema(s) for DataFrame results.
"""
from pandera import (
    DataFrameSchema,
    Column,
    Check,
    PandasDtype,
)

describe_df_schema = DataFrameSchema(
    columns={
        "geometry": Column(
            pandas_dtype=PandasDtype.String,
            checks=None,
            nullable=False,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "X": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Y": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "I": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "E": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "C - C": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "C - I": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "I - I": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "C - E": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "I - E": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "E - E": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law vs. lognormal R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law vs. lognormal p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law vs. exponential R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law vs. exponential p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lognormal vs. exponential R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lognormal vs. exponential p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law vs. truncated_power_law R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law vs. truncated_power_law p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace exponential Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lognormal Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace truncated_power_law Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law alpha": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law exponent": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law cut-off": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace power_law sigma": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lognormal sigma": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lognormal mu": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace exponential lambda": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace truncated_power_law lambda": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace truncated_power_law alpha": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace truncated_power_law exponent": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lognormal loglikelihood": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace exponential loglikelihood": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace truncated_power_law loglikelihood": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law vs. lognormal R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law vs. lognormal p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law vs. exponential R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law vs. exponential p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lognormal vs. exponential R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lognormal vs. exponential p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law vs. truncated_power_law R": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law vs. truncated_power_law p": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch exponential Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lognormal Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch truncated_power_law Kolmogorov-Smirnov distance D": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law alpha": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law exponent": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law cut-off": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch power_law sigma": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lognormal sigma": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lognormal mu": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch exponential lambda": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch truncated_power_law lambda": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch truncated_power_law alpha": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch truncated_power_law exponent": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lognormal loglikelihood": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch exponential loglikelihood": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch truncated_power_law loglikelihood": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[],
            nullable=True,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Number of Traces": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Number of Branches": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Fracture Intensity B21": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Fracture Intensity P21": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Areal Frequency P20": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Areal Frequency B20": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Trace Mean Length": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Branch Mean Length": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Dimensionless Intensity P22": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Dimensionless Intensity B22": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Connections per Trace": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Connections per Branch": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=2.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Fracture Intensity (Mauldon)": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Fracture Density (Mauldon)": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Trace Mean Length (Mauldon)": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "Connection Frequency": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "trace lengths cut off proportion": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "branch lengths cut off proportion": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "name": Column(
            pandas_dtype=PandasDtype.String,
            checks=None,
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
        "radius": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=26.0),
            ],
            nullable=False,
            allow_duplicates=True,
            coerce=True,
            required=True,
            regex=False,
        ),
    },
)
