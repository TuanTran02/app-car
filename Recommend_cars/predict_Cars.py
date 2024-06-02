import streamlit as st
import numpy as np
import pandas as pd
import base64
import requests
import joblib

def app():
    # st.header('Dự đoán')
    def sidebar_bg(img_url):
        side_bg_ext = 'jpg'  # Assuming the image format is PNG (can be adjusted if needed)

        # Retrieve image data from URL
        response = st.cache_resource(requests.get)(img_url, stream=True)
        img_data = response.content

        # Encode image data as base64
        encoded_data = base64.b64encode(img_data).decode()

        # Apply background image style to sidebar
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] > div:first-child {{
                background: url(data:image/{side_bg_ext};base64,{encoded_data});
                background-size: cover;  /* Adjust background sizing as needed */
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Example usage with a valid image URL
    # img_url = "https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTEwL3Jhd3BpeGVsb2ZmaWNlM19taW5pbWFsX2ZsYXRfdmVjdG9yX2Flc3RoZXRpY19pbGx1c3RyYXRpb25fb2ZfYV9hYWMyODk1Ny02ODI3LTQ3OGUtOTQ2Ni0wNWI0MzVhYjk2MmQtYi5qcGc.jpg"  # Replace with your desired image URL
    img_url = "https://i.pinimg.com/originals/28/c1/d7/28c1d768683e896a84a4ec5f37f02463.jpg"  # Replace with your desired image URL
    sidebar_bg(img_url)
    
    with st.sidebar:
        message_html = """
        <style>
            .recommend-message {
                color: white; /* Change this to your desired color */
                font-weight: bold;
                font-size: 45px;
            }
        </style>

        <div class="recommend-message">PREDICT</div>
        """
        st.sidebar.write(message_html, unsafe_allow_html=True)

    # Load models and scalers
    model = joblib.load('C:\Python\Capstone\Recommend_cars\Model\knn_model2.joblib')
    encoder = joblib.load('C:\Python\Capstone\Recommend_cars\Model\encoder2.joblib')
    scaler = joblib.load('C:\Python\Capstone\Recommend_cars\Model\scaler2.joblib')

    # Creating two columns
    col1, col2 = st.columns(2)

    # Column for design and style
    with col1:
        # Danh sách các hãng xe
        brands = ['KIA', 'TOYOTA', 'FORD', 'MAZDA', 'BMW', 'HONDA', 'NISSAN',
                'HYUNDAI', 'MITSUBISHI', 'VOLKSWAGEN', 'MG', 'LEXUS', 'CHEVROLET',
                'LANDROVER', 'VINFAST', 'MERCEDES-BENZ', 'SUZUKI', 'PORSCHE',
                'THACO', 'ISUZU', 'DAEWOO', 'SUBARU', 'AUDI', 'PEUGEOT', 'VOLVO',
                'BENTLEY', 'HAVAL', 'JEEP', 'ROVER', 'MINI', 'ROLLS ROYCE']
        # Sắp xếp danh sách theo thứ tự bảng chữ cái
        sorted_brands = sorted(brands)
        # Sử dụng danh sách đã sắp xếp trong selectbox
        brand = st.selectbox('Hãng xe', options=sorted_brands)

        # Dòng xe theo từng hãng        
        models = {
            'KIA': {'I10': {'body_styles': ['Hatchback'], 'engine_sizes': [1.0, 1.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                    'Seltos': {'body_styles': ['SUV'], 'engine_sizes': [1.4, 1.6], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                    'Cerato': {'body_styles': ['Sedan'], 'engine_sizes': [1.6, 2.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                    'Sorento': {'body_styles': ['SUV'], 'engine_sizes': [2.2, 2.4], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Dầu', 'Xăng']},
                    'Carnival': {'body_styles': ['Wagon'], 'engine_sizes': [2.2, 3.5], 'seats': [7, 8], 'transmissions': ['Số tự động'], 'fuels': ['Dầu', 'Xăng']},
                    'K3': {'body_styles': ['Sedan'], 'engine_sizes': [1.6, 2.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'TOYOTA': {'Camry': {'body_styles': ['Sedan'], 'engine_sizes': [2.0, 2.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                       'Rush': {'body_styles': ['SUV'], 'engine_sizes': [1.5], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                       'Vios': {'body_styles': ['Sedan'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'Fortuner': {'body_styles': ['SUV'], 'engine_sizes': [2.4, 2.7], 'seats': [7], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Dầu', 'Xăng']},
                       'Innova': {'body_styles': ['Wagon'], 'engine_sizes': [2.0], 'seats': [7], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'Corolla Altis': {'body_styles': ['Sedan'], 'engine_sizes': [1.8, 2.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng', 'Hybrid']},
                       'Yaris': {'body_styles': ['Hatchback'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'FORD': {'Ranger': {'body_styles': ['Bán tải'], 'engine_sizes': [2.0, 2.2, 3.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Dầu']},
                     'Everest': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 2.2], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Dầu']},
                     'EcoSport': {'body_styles': ['SUV'], 'engine_sizes': [1.0, 1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                     'Explorer': {'body_styles': ['SUV'], 'engine_sizes': [2.3], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                     'Transit': {'body_styles': ['Van'], 'engine_sizes': [2.2], 'seats': [16], 'transmissions': ['Số sàn'], 'fuels': ['Dầu']},
                     'Fiesta': {'body_styles': ['Hatchback'], 'engine_sizes': [1.0, 1.5], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'MAZDA': {'CX-5': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 2.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'CX-8': {'body_styles': ['SUV'], 'engine_sizes': [2.5], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'Mazda3': {'body_styles': ['Sedan', 'Hatchback'], 'engine_sizes': [1.5, 2.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                      'Mazda6': {'body_styles': ['Sedan'], 'engine_sizes': [2.0, 2.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'BT-50': {'body_styles': ['Bán tải'], 'engine_sizes': [2.2, 3.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Dầu']},
                      'MX-5': {'body_styles': ['Convertible/Cabriolet'], 'engine_sizes': [2.0], 'seats': [2], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'BMW': {'Series 3': {'body_styles': ['Sedan'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                    'Series 5': {'body_styles': ['Sedan'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                    'Series 7': {'body_styles': ['Sedan'], 'engine_sizes': [3.0, 4.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                    'X1': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                    'X3': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                    'X5': {'body_styles': ['SUV'], 'engine_sizes': [3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                    'X7': {'body_styles': ['SUV'], 'engine_sizes': [3.0, 4.4], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'HONDA': {'City': {'body_styles': ['Sedan'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                      'Civic': {'body_styles': ['Sedan'], 'engine_sizes': [1.5, 1.8], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                      'CR-V': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 2.4], 'seats': [5, 7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'HR-V': {'body_styles': ['SUV'], 'engine_sizes': [1.8], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'Jazz': {'body_styles': ['Hatchback'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'Brio': {'body_styles': ['Hatchback'], 'engine_sizes': [1.2], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'Accord': {'body_styles': ['Sedan'], 'engine_sizes': [1.5, 2.0, 2.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'NISSAN': {'Navara': {'body_styles': ['Bán tải'], 'engine_sizes': [2.5], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Dầu']},
                       'Terra': {'body_styles': ['SUV'], 'engine_sizes': [2.5], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Dầu']},
                       'Sunny': {'body_styles': ['Sedan'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'X-Trail': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 2.5], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                       'Juke': {'body_styles': ['SUV'], 'engine_sizes': [1.6], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'HYUNDAI': {'i10': {'body_styles': ['Hatchback'], 'engine_sizes': [1.0, 1.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                        'i20': {'body_styles': ['Hatchback'], 'engine_sizes': [1.2, 1.4], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                        'i30': {'body_styles': ['Hatchback'], 'engine_sizes': [1.6, 2.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                        'Tucson': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 2.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        'SantaFe': {'body_styles': ['SUV'], 'engine_sizes': [2.2, 2.4], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Dầu', 'Xăng']},
                        'Elantra': {'body_styles': ['Sedan'], 'engine_sizes': [1.6, 2.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'MITSUBISHI': {'Xpander': {'body_styles': ['MPV'], 'engine_sizes': [1.5], 'seats': [7], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                           'Outlander': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 2.4], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                           'Pajero Sport': {'body_styles': ['SUV'], 'engine_sizes': [2.4, 3.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Dầu', 'Xăng']},
                           'Attrage': {'body_styles': ['Sedan'], 'engine_sizes': [1.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                           'Mirage': {'body_styles': ['Hatchback'], 'engine_sizes': [1.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'VOLKSWAGEN': {'Polo': {'body_styles': ['Sedan', 'Hatchback'], 'engine_sizes': [1.2, 1.6], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                           'Tiguan': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                           'Touareg': {'body_styles': ['SUV'], 'engine_sizes': [3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                           'Passat': {'body_styles': ['Sedan'], 'engine_sizes': [1.8, 2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'MG': {'ZS': {'body_styles': ['SUV'], 'engine_sizes': [1.0, 1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                   'HS': {'body_styles': ['SUV'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                   '5': {'body_styles': ['Sedan'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                   '6': {'body_styles': ['Sedan'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'LEXUS': {'RX': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'NX': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'ES': {'body_styles': ['Sedan'], 'engine_sizes': [2.5, 3.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'LS': {'body_styles': ['Sedan'], 'engine_sizes': [3.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'GX': {'body_styles': ['SUV'], 'engine_sizes': [4.6], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'CHEVROLET': {'Colorado': {'body_styles': ['Bán tải'], 'engine_sizes': [2.5, 2.8], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Dầu']},
                          'Trax': {'body_styles': ['SUV'], 'engine_sizes': [1.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                          'Captiva': {'body_styles': ['SUV'], 'engine_sizes': [2.4], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                          'Spark': {'body_styles': ['Hatchback'], 'engine_sizes': [1.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                          'Aveo': {'body_styles': ['Sedan'], 'engine_sizes': [1.4], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'LANDROVER': {'Range Rover': {'body_styles': ['SUV'], 'engine_sizes': [3.0, 4.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                          'Discovery': {'body_styles': ['SUV'], 'engine_sizes': [3.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                          'Defender': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.0], 'seats': [5, 7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                          'Evoque': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'VINFAST': {'Lux A2.0': {'body_styles': ['Sedan'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        'Lux SA2.0': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        'Fadil': {'body_styles': ['Hatchback'], 'engine_sizes': [1.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        'President': {'body_styles': ['SUV'], 'engine_sizes': [6.2], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'MERCEDES-BENZ': {'C-Class': {'body_styles': ['Sedan'], 'engine_sizes': [1.5, 2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                              'E-Class': {'body_styles': ['Sedan'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                              'S-Class': {'body_styles': ['Sedan'], 'engine_sizes': [3.0, 4.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                              'GLC': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                              'GLE': {'body_styles': ['SUV'], 'engine_sizes': [3.0], 'seats': [5, 7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                              'GLS': {'body_styles': ['SUV'], 'engine_sizes': [3.0, 4.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'SUZUKI': {'Swift': {'body_styles': ['Hatchback'], 'engine_sizes': [1.2], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'Ciaz': {'body_styles': ['Sedan'], 'engine_sizes': [1.4], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'Ertiga': {'body_styles': ['MPV'], 'engine_sizes': [1.5], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                       'XL7': {'body_styles': ['SUV'], 'engine_sizes': [1.5], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'PORSCHE': {'Cayenne': {'body_styles': ['SUV'], 'engine_sizes': [3.0, 4.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                        'Macan': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        'Panamera': {'body_styles': ['Sedan'], 'engine_sizes': [2.9, 4.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                        '911': {'body_styles': ['Coupe'], 'engine_sizes': [3.0, 3.8], 'seats': [2, 4], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'THACO': {'Bus': {'body_styles': ['Bus'], 'engine_sizes': [4.0, 6.0], 'seats': [30, 45], 'transmissions': ['Số sàn'], 'fuels': ['Dầu']},
                      'Truck': {'body_styles': ['Truck'], 'engine_sizes': [4.0, 5.0], 'seats': [2], 'transmissions': ['Số sàn'], 'fuels': ['Dầu']},
                      'Mighty': {'body_styles': ['Truck'], 'engine_sizes': [3.5, 4.0], 'seats': [2], 'transmissions': ['Số sàn'], 'fuels': ['Dầu']}},
            'ISUZU': {'D-Max': {'body_styles': ['Bán tải'], 'engine_sizes': [1.9, 3.0], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Dầu']},
                      'Mu-X': {'body_styles': ['SUV'], 'engine_sizes': [1.9, 3.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Dầu']}},
            'DAEWOO': {'Lacetti': {'body_styles': ['Sedan'], 'engine_sizes': [1.6], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'Nubira': {'body_styles': ['Sedan'], 'engine_sizes': [1.6], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                       'Gentra': {'body_styles': ['Sedan'], 'engine_sizes': [1.5], 'seats': [5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']}},
            'SUBARU': {'Forester': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                       'Outback': {'body_styles': ['SUV'], 'engine_sizes': [2.5], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                       'XV': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'AUDI': {'A4': {'body_styles': ['Sedan'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                     'A6': {'body_styles': ['Sedan'], 'engine_sizes': [2.0, 3.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                     'Q3': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                     'Q5': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                     'Q7': {'body_styles': ['SUV'], 'engine_sizes': [3.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'PEUGEOT': {'3008': {'body_styles': ['SUV'], 'engine_sizes': [1.6], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        '5008': {'body_styles': ['SUV'], 'engine_sizes': [1.6], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                        '2008': {'body_styles': ['SUV'], 'engine_sizes': [1.2], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'VOLVO': {'XC40': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'XC60': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'XC90': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'S90': {'body_styles': ['Sedan'], 'engine_sizes': [2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'BENTLEY': {'Bentayga': {'body_styles': ['SUV'], 'engine_sizes': [4.0, 6.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                        'Continental': {'body_styles': ['Coupe'], 'engine_sizes': [4.0, 6.0], 'seats': [4], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                        'Flying Spur': {'body_styles': ['Sedan'], 'engine_sizes': [4.0, 6.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'HAVAL': {'H6': {'body_styles': ['SUV'], 'engine_sizes': [1.5, 2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                      'H9': {'body_styles': ['SUV'], 'engine_sizes': [2.0], 'seats': [7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'JEEP': {'Wrangler': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.6], 'seats': [4, 5], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                     'Cherokee': {'body_styles': ['SUV'], 'engine_sizes': [2.0, 3.2], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                     'Compass': {'body_styles': ['SUV'], 'engine_sizes': [2.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'ROVER': {'Range Rover': {'body_styles': ['SUV'], 'engine_sizes': [3.0, 4.4], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']},
                      'Range Rover Sport': {'body_styles': ['SUV'], 'engine_sizes': [3.0, 5.0], 'seats': [5, 7], 'transmissions': ['Số tự động'], 'fuels': ['Xăng', 'Hybrid']}},
            'MINI': {'Cooper': {'body_styles': ['Hatchback'], 'engine_sizes': [1.5, 2.0], 'seats': [4], 'transmissions': ['Số tự động', 'Số sàn'], 'fuels': ['Xăng']},
                     'Countryman': {'body_styles': ['SUV'], 'engine_sizes': [1.5, 2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                     'Clubman': {'body_styles': ['Hatchback'], 'engine_sizes': [1.5, 2.0], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}},
            'ROLLS ROYCE': {'Ghost': {'body_styles': ['Sedan'], 'engine_sizes': [6.6], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                            'Phantom': {'body_styles': ['Sedan'], 'engine_sizes': [6.8], 'seats': [5], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']},
                            'Wraith': {'body_styles': ['Coupe'], 'engine_sizes': [6.6], 'seats': [4], 'transmissions': ['Số tự động'], 'fuels': ['Xăng']}}
        }

        if brand in models:
            model_car = st.selectbox('Dòng xe', options=list(models[brand].keys()))
            body_style_options = models[brand][model_car]['body_styles']
            body_style = st.selectbox('Kiểu dáng', options=body_style_options)
        else:
            model_car = st.selectbox('Dòng xe', options=['No models available'])
            body_style = st.selectbox('Kiểu dáng', options=['No styles available'])
        
        # body_style = st.selectbox('Kiểu dáng', options=['Sedan', 'SUV', 'Hatchback', 'Wagon', 'Convertible/Cabriolet', 'Coupe'])
        color = st.selectbox('Màu sắc', options=['Trắng', 'Đen', 'Bạc', 'Xanh', 'Đỏ', 'Vàng'])
        # transmission = st.selectbox('Hộp số', options=['Số tự động', 'Số sàn'])
        # fuel = st.selectbox('Nhiên liệu', options=['Xăng', 'Dầu', 'Hybrid', 'Điện'])
        year = st.slider('Năm sản xuất', min_value=2000, max_value=2024, value=2023)
    # Column for technical specifications
    with col2:
        if brand in models:
            engine_size_options = models[brand][model_car]['engine_sizes']
            engine = st.selectbox('Dung tích động cơ', options=engine_size_options)
            seats_options = models[brand][model_car]['seats']
            seats = st.selectbox('Số chỗ ngồi', options=seats_options)
            transmission_options = models[brand][model_car]['transmissions']
            transmission = st.selectbox('Hộp số', options=transmission_options)
            fuel_options = models[brand][model_car]['fuels']
            fuel = st.selectbox('Nhiên liệu', options=fuel_options)
        else:
            engine = st.selectbox('Dung tích động cơ', options=['No engine sizes available'])
            seats = st.selectbox('Số chỗ ngồi', options=['No seats available'])
            transmission = st.selectbox('Hộp số', options=['No transmission available'])
            fuel = st.selectbox('Nhiên liệu', options=['No fuel available'])
        km = st.slider('Km đã đi', min_value=0, max_value=500000, value=50000, step=1000)

    # Predict button in the center or under columns
    if st.button('Dự đoán giá'):
        # Encode and predict
        features = np.array([[body_style, color, transmission, fuel,brand]])
        encoded_features = encoder.transform(features)
        features = np.array([[year, seats, km, engine]])
        all_features = np.hstack((features, encoded_features))
        all_features = scaler.transform(all_features)
        prediction = model.predict(all_features)
        st.write(f'Dự đoán giá xe: {prediction[0]:,.0f} VND')


