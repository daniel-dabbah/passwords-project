import streamlit as st


page1 = st.Page("page1.py", title="page1")
page2 = st.Page("page2.py", title="page2")
page3 = st.Page("page3.py", title="page3")
page4 = st.Page("page4.py", title="page4")
page5 = st.Page("page5.py", title="page5")
page6 = st.Page("page6.py", title="page6")


pg = st.navigation([page1, page2, page3, page4, page5, page6])
pg.run()

st.markdown(
    """
<style>
body, html {
    direction: LTR;
    unicode-bidi: bidi-override;
    text-align: right;
}
</style>
""",
    unsafe_allow_html=True,
)