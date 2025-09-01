import streamlit as st
import pandas as pd
from src.serving.serving import load_model, predict_single, predict_batch


def main():
    st.title("Ham/Spam App Classification")

    tab1, tab2 = st.tabs([
        "Single-Prediction",
        "Batch-Prediction",
    ])


    with tab1:
        st.subheader("Single Prediction")
        text_input = st.text_input("Enter text for prediction")

        if st.button("Predict (Single)"):
            wrapper = load_model()
            try:
                result = predict_single(text_input, wrapper)
                st.success(f"Result: {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


    with tab2:
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload batch file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            df = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith("csv")
                else pd.read_excel(uploaded_file)
            )
            st.write("Preview:")
            st.dataframe(df.head())

            if st.button("Predict (Batch)"):
                wrapper = load_model()
                try:
                    result_df = predict_batch(df, wrapper, text_column="message")
                    st.dataframe(result_df)
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")


if __name__ == "__main__":
    main()
