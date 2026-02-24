def apply_dynamic_filters(df, filters: dict):
    filtered = df.copy()

    for column, condition in filters.items():

        if column not in filtered.columns:
            continue

        # Range filter
        if isinstance(condition, dict):
            if "min" in condition and condition["min"] is not None:
                filtered = filtered[filtered[column] >= condition["min"]]

            if "max" in condition and condition["max"] is not None:
                filtered = filtered[filtered[column] <= condition["max"]]

        # Boolean filter
        elif isinstance(condition, bool):
            filtered = filtered[filtered[column] == int(condition)]

        # Exact match
        else:
            filtered = filtered[filtered[column] == condition]

    return filtered