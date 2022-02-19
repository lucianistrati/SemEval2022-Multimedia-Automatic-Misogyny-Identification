import xai.data

from data.TRAINING_csvs.training_splitter import load_for_explainability
labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
text_column = "Text Transcription"

for label_column in labels_columns:
    data = load_for_explainability(label_column)
    df = data

    ims = xai.imbalance_plot(df, label_column)

    im = xai.imbalance_plot(df, label_column)

    bal_df = xai.balance(df, label_column, upsample=0.8)

    groups = xai.group_by_columns(df, [label_column])
    for group, group_df in groups:
        print(group)
        print(group_df[label_column].head(), "\n")


    _ = xai.correlations(df, include_categorical=True, plot_type="matrix")

    _ = xai.correlations(df, include_categorical=True)

    x, y = data[text_column].to_list(), data[label_column].to_list()

    categorical_cols = [label_column]
    # Balanced train-test split with minimum 300 examples of
    #     the cross of the target y and the column gender

    x_train, y_train, x_test, y_test, train_idx, test_idx = \
        xai.balanced_train_test_split(
                x, y, "gender",
                min_per_group=300,
                max_per_group=300,
                categorical_cols=categorical_cols)

    x_train_display = bal_df[train_idx]
    x_test_display = bal_df[test_idx]

    print("Total number of examples: ", x_test.shape[0])

    df_test = x_test_display.copy()
    df_test["loan"] = y_test

    _= xai.imbalance_plot(df_test, "gender", "loan", categorical_cols=categorical_cols)


    model = build_model(proc_df.drop("loan", axis=1))

    model.fit(f_in(x_train), y_train, epochs=50, batch_size=512)

    probabilities = model.predict(f_in(x_test))
    predictions = list((probabilities >= 0.5).astype(int).T[0])


    def get_avg(x, y):
        return model.evaluate(f_in(x), y, verbose=0)[1]

    imp = xai.feature_importance(x_test, y_test, get_avg)

    imp.head()

    _= xai.metrics_plot(
            y_test,
            probabilities)

    _ = xai.metrics_plot(
        y_test,
        probabilities,
        df=x_test_display,
        cross_cols=["gender"],
        categorical_cols=categorical_cols)

    _ = xai.metrics_plot(
        y_test,
        probabilities,
        df=x_test_display,
        cross_cols=["gender", "ethnicity"],
        categorical_cols=categorical_cols)

    xai.confusion_matrix_plot(y_test, pred)

    _ = xai.roc_plot(y_test, probabilities)


    protected = ["gender", "ethnicity", "age"]
    _ = [xai.roc_plot(
        y_test,
        probabilities,
        df=x_test_display,
        cross_cols=[p],
        categorical_cols=categorical_cols) for p in protected]

    d = xai.smile_imbalance(
        y_test,
        probabilities)


    d = xai.smile_imbalance(
        y_test,
        probabilities,
        display_breakdown=True)

    d = xai.smile_imbalance(
        y_test,
        probabilities,
        bins=9,
        threshold=0.75,
        manual_review=0.375,
        display_breakdown=False)