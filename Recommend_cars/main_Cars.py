import streamlit as st
from streamlit_option_menu import option_menu

import about_Cars, compare_Cars, predict_Cars, recommend_Cars

st.set_page_config(
    page_title="Recommend Cars",
    page_icon="🚘",
    layout="wide"
)

class MultiApp:

    def __init__(self):
        self.apps = []

    # def add_app(self, title, func):

    #     self.apps.append({
    #         "title": title,
    #         "function": func
    #     })

    def run():
        app = option_menu(
            menu_title = "MAIN MENU", 
            options = ["Recommend", "Predict", "Compare", "About"],
            icons = ["car-front", "calculator", "search", "info-circle-fill"],
            default_index = 0,
            orientation = "horizontal",
            styles={
                            "container": {"padding": "5!important","background-color":'white'},
                "icon": {"color": "black", "font-size": "23px"}, 
                "nav-link": {"color":"black","font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#BDE4F5"},
                "nav-link-selected": {"background-color": "#2BB2EF"},}
            )

        if app == "Recommend":
            recommend_Cars.app()
        if app == "Predict":
            predict_Cars.app()
        if app == "Compare":
            compare_Cars.app()
        if app == "About":
            about_Cars.app()

    run()