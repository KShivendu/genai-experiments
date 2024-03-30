# Usage: streamlit run main.py

import streamlit as st
from engine import search


def handle_query():
    query_str = st.session_state.query
    records = search(query_str)

    if len(records) > 0:
        cols = st.columns(5)
        for col, record in zip(cols, records):
            with col:
                caption = f"Payload: {record.payload} | Score: {record.score}"

                st.image(
                    record.payload["image_path"],
                    caption=caption,
                    use_column_width=True,
                )

    st.session_state.results = records


def main():
    st.title("Multi-modal reranking search")
    st.text_input("Enter your query:", key="query", on_change=handle_query)
    st.button("Search results", on_click=handle_query)

if __name__ == "__main__":
    main()
