import numpy as np
import streamlit as st
import pandas as pd
from num2words import num2words
import base64
import requests

def app():

    # st.header('Đề xuất')

    df = pd.read_excel('Recommend_cars/Data/Data_Recommend.xlsx')

    col1 = st.sidebar
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
    # img_url = "https://4kwallpapers.com/images/wallpapers/porsche-918-spyder-1290x2796-13004.jpg" 
    img_url = "https://i.pinimg.com/originals/09/22/ad/0922ad89888ffd9968858717c1901ac1.jpg"
    sidebar_bg(img_url)

    with col1:
        # Widget để chọn Tình Trạng xe
        selected_state = col1.multiselect("Chọn Tình trạng xe:", options=df['Tình trạng'].unique())

        if selected_state:
                filtered_brands = df.loc[(df['Tình trạng'].isin(selected_state)), 'Hãng xe'].unique()
        else:
            filtered_brands = df['Hãng xe'].unique()
        # Sắp xếp Hãng xe theo bảng chữ cái
        filtered_brands = sorted(filtered_brands)

        selected_brands = st.multiselect("Chọn Hãng xe:", filtered_brands)
        # Lọc Dòng xe dựa trên Hãng xe đã chọn
        filtered_vehicles = df.loc[df['Hãng xe'].isin(selected_brands), 'Dòng xe'].unique()
        # Sắp xếp theo chữ cái Dòng xe
        
        # Widget để chọn Dòng xe
        selected_vehicles = st.multiselect("Chọn Dòng xe:", filtered_vehicles)
        
        if selected_brands:
            if selected_vehicles:
                # Lọc Kiểu dáng dựa trên Dòng xe đã chọn
                filtered_styles = df.loc[(df['Dòng xe'].isin(selected_vehicles)), 'Kiểu dáng'].unique()
            else:
                # Lọc Kiểu dáng dựa trên Hãng xe đã chọn
                filtered_styles = df.loc[(df['Hãng xe'].isin(selected_brands)), 'Kiểu dáng'].unique()
        else:
            filtered_styles = df['Kiểu dáng'].unique()

        # Sắp xếp theo Kiểu dáng
        filtered_styles = sorted(filtered_styles)
        # Widget để chọn Kiểu dáng
        selected_styles = st.multiselect("Chọn Kiểu dáng:", filtered_styles)
        # Widget để chọn Năm sx
        if 'Xe Cũ' in selected_state or 'Xe cũ' in selected_state:
            if selected_brands:
                if selected_vehicles:
                    # Lọc Năm sản xuất dựa trên Dòng xe đã chọn
                    filtered_year = df.loc[df['Dòng xe'].isin(selected_vehicles), 'Năm SX'].unique()
                else:
                    # Lọc Năm sản xuất dựa trên Hãng xe đã chọn
                    filtered_year = df.loc[df['Hãng xe'].isin(selected_brands), 'Năm SX'].unique()
            else:
                filtered_year = df['Năm SX'].unique()
        else:
            filtered_year = df['Năm SX'].unique()

        # Sắp xếp các năm sản xuất theo thứ tự tăng dần
        filtered_year = sorted(filtered_year, reverse=True)

        selected_year = st.multiselect("Chọn Năm sản xuất:", filtered_year)

        
        # Widget để chọn Nhiên liệu 
        if selected_brands:
            if selected_vehicles:
            # Lọc Kiểu dáng dựa trên Dòng xe đã chọn
                filtered_fuel = df.loc[(df['Dòng xe'].isin(selected_vehicles)), 'Nhiên liệu'].unique()
            else:
                # Lọc Kiểu dáng dựa trên Hãng xe đã chọn
                filtered_fuel = df.loc[(df['Hãng xe'].isin(selected_brands)), 'Nhiên liệu'].unique()
        else:
            filtered_fuel = df['Nhiên liệu'].unique()
        selected_segment = st.multiselect("Chọn Nhiên liệu:", filtered_fuel)
        
        # Widget để chọn Hộp số 
        if selected_brands:
            if selected_vehicles:
            # Lọc Kiểu dáng dựa trên Dòng xe đã chọn
                filtered_hopso = df.loc[(df['Dòng xe'].isin(selected_vehicles)), 'Hộp số'].unique()
            else:
                # Lọc Kiểu dáng dựa trên Hãng xe đã chọn
                filtered_hopso = df.loc[(df['Hãng xe'].isin(selected_brands)), 'Hộp số'].unique()
        else:
            filtered_hopso = df['Hộp số'].unique()
        selected_hopso = st.multiselect("Chọn Hộp số:", filtered_hopso)
        
        # Widget để chọn Số chỗ ngồi
        if selected_brands:
            if selected_vehicles:
            # Lọc Kiểu dáng dựa trên Dòng xe đã chọn
                filtered_seat = df.loc[(df['Dòng xe'].isin(selected_vehicles)), 'Số chỗ ngồi'].unique()
            else:
                # Lọc Kiểu dáng dựa trên Hãng xe đã chọn
                filtered_seat = df.loc[(df['Hãng xe'].isin(selected_brands)), 'Số chỗ ngồi'].unique()
        else:
            filtered_seat = df['Số chỗ ngồi'].unique()
        # Sắp xếp số chỗ ngồi theo thứ tự tăng dần
        filtered_seat = sorted(filtered_seat)

        selected_seat = st.multiselect("Chọn Số chỗ ngồi:", filtered_seat)

        # Widget để chọn Màu sắc
        if selected_brands:
            if selected_vehicles:
            # Lọc Kiểu dáng dựa trên Dòng xe đã chọn
                filtered_color = df.loc[(df['Dòng xe'].isin(selected_vehicles)), 'Màu sắc'].unique()
            else:
                # Lọc Kiểu dáng dựa trên Hãng xe đã chọn
                filtered_color = df.loc[(df['Hãng xe'].isin(selected_brands)), 'Màu sắc'].unique()
        else:
            filtered_color = df['Màu sắc'].unique()
        selected_color = st.multiselect("Chọn Màu sắc:", filtered_color)

        # Widget để chọn Đơn giá
        if selected_brands:
            if selected_vehicles:
            # Lọc Kiểu dáng dựa trên Dòng xe đã chọn
                filtered_price = df.loc[(df['Dòng xe'].isin(selected_vehicles)), 'Khoảng giá'].unique()
            else:
                # Lọc Kiểu dáng dựa trên Hãng xe đã chọn
                filtered_price = df.loc[(df['Hãng xe'].isin(selected_brands)), 'Khoảng giá'].unique()
        else:
            filtered_price = df['Khoảng giá'].unique()
        # Sắp xếp các Giá tiền theo thứ tự tăng dần
        filtered_price = ['0 - 500 triệu', '500 triệu - 1 tỷ', '1 tỷ - 1.5 tỷ', '1.5 tỷ - 2 tỷ', '2 tỷ trở lên']

        selected_price = st.multiselect("Chọn Giá tiền:", filtered_price)
        

    # Nút lọc
        filter_button = st.button("Tìm kiếm")
    
    # if selected_brands or selected_vehicles or selected_styles or selected_state or selected_segment or selected_hopso or selected_seat or selected_year: #or (selected_km_min and selected_km_max):
#     # Khởi tạo điều kiện lọc
    mask = pd.Series([True] * len(df))

    
    # Lọc DataFrame dựa trên các lựa chọn
    if selected_state:
        mask &= df['Tình trạng'].isin(selected_state)
    if selected_brands:
        mask &= df['Hãng xe'].isin(selected_brands)
    if selected_vehicles:
        mask &= df['Dòng xe'].isin(selected_vehicles)
    if selected_styles:
        mask &= df['Kiểu dáng'].isin(selected_styles)
    if selected_segment:
        mask &= df['Nhiên liệu'].isin(selected_segment)
    if selected_hopso:
        mask &= df['Hộp số'].isin(selected_hopso)
    if selected_seat:
        mask &= df['Số chỗ ngồi'].isin(selected_seat)
    if selected_year:
        mask &= df['Năm SX'].isin(selected_year)
    if selected_color:
        mask &= df['Màu sắc'].isin(selected_color)
    if selected_price:
        mask &= df['Khoảng giá'].isin(selected_price)
    
    if selected_brands or selected_vehicles or selected_styles or selected_state or selected_segment or selected_hopso or selected_seat or selected_year or selected_color or selected_price:
        filter_button_2 = st.success(str(selected_state + selected_brands + selected_vehicles + selected_styles + selected_year + selected_segment + selected_hopso + selected_seat + selected_color + selected_price))
    else:
        st.write("Hãy lựa chọn tiêu chí phù hợp với bạn !")

    # Nếu nút lọc được nhấn
    if filter_button:
    
        filtered_df = df[mask]
        
        # Reset index
        filtered_df = filtered_df.reset_index(drop=True)
        
        if filtered_df.shape[0] == 0:
            st.error("***Kết quả không được tìm thấy***\n\n Hãy chọn 🔻 / nhập lại 💬    🧐")
        else: 
            # Hiển thị số lượng features
            st.write("Số lượng xe được tìm thấy: ", filtered_df.shape[0])
            

            # Hiển thị hình ảnh, tên xe và đơn giá
            cols = st.columns(3)
            count = 0
            for index, row in filtered_df.iterrows():
                with cols[index % 3]:
                    if 'Image_URL' in df.columns:
                        st.image(row['Image_URL'], caption=row['Tên Xe'])
                        # st.image(row['Image_URL'])
                        # st.write(f"""
                        #             <div style="text-align: center;">
                        #             <font-size: 10px;'> <strong>{str(row['Tên Xe'])}</strong>
                        #             </div>
                        #             """, unsafe_allow_html=True)

                    # st.write(f"<div style='text-align: center;'>Đơn giá: {str(row['Đơn giá'])}</div>", unsafe_allow_html=True)
                        if not pd.isna(row['Đơn giá']):                              
                            if isinstance(row['Đơn giá'], float) or isinstance(row['Đơn giá'], int):
                                number = row['Đơn giá']
                                formatted_number = f"{number:,.0f}₫"
                                st.write(f"""
                                        <div style="text-align: center;">
                                        <font-size: 10px;'>Giá: <strong>{formatted_number}</strong>
                                        </div>
                                        """, unsafe_allow_html=True)                                    
                                # price_in_words = row['Giá bằng chữ']
                                # price_text = f"""
                                #     <div style="text-align: center;">
                                #     <font size="10px">Bằng chữ: <strong>{price_in_words}</strong></font>
                                #     </div>
                                # """
                                
                                # st.write(price_text, unsafe_allow_html=True)
                                
                            else:
                                st.write(f"""
                                    <div style="text-align: center;">
                                    <font-size: 10px;'>Giá: <strong>{str(row['Đơn giá'])}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    
                    col1, col2 = st.columns(2)  # Chia thành 2 cột
                    with col1:
                        st.caption('- ' + str(row['Hộp số']))
                        st.caption('- ' + str(row['Nhiên liệu']))
                        st.caption('- ' + str(row['Tỉnh thành']))
                        st.caption('- ' + str(row['Kiểu dáng']))
                    with col2:                            
                        if row['Tình trạng'] == 'Xe Mới' or row['Tình trạng'] == 'Xe mới':
                            st.caption('- ' + str(row['Tình trạng']))
                        else: 
                            if np.isnan(row['Km đã đi']):
                                st.caption('- ' + str(row['Tình trạng']))
                            else:
                                st.caption('- ' + str(row['Tình trạng']) + ' (' + str(int(row['Km đã đi'])) + " KM" + ')')
                        st.caption('- ' + str(row['Số chỗ ngồi']) + ' chỗ')
                        st.caption('- ' + str(row['Năm SX']))
                        st.caption('- ' + str(row['Xuất xứ']))
                                    
                    # st.write('')  # Thêm một dòng trống để tạo khoảng cách                        
                    col1, col2 = st.columns(2)  # Chia thành 2 cột
                    with col2:
                        # Button with bold text using markdown syntax
                        button_content = f"**Xem chi tiết**"

                        # Button styling using inline CSS
                        button_style = """
                        <style>
                            .buy-button {
                            background-color: #CBEAF5;
                            color: white;
                            padding: 5px 10px;
                            border: none;
                            border-radius: 10px;
                            float: right;
                            cursor: pointer;
                            text-decoration: none;
                            font-weight: normal !important;
                            }
                        </style>
                        """
                        # Combine button content, link, and style
                        content = f"""{button_style}
                        <a class="buy-button" href="{row['Link xe']}" target="_blank">{button_content}</a>
                        """
                        # Display content with unsafe_allow_html for button styling
                        st.markdown(content, unsafe_allow_html=True)
                    
                    # with col1:                            
                    #     detail_button = st.button(str(count) + ". Xem chi tiết")
                    #     count += 1
                    #     if detail_button:
                    #         with st.expander("Nội dung chi tiết", expanded=True):
                    #             st.write("Đây là nội dung chi tiết")
                        
                    
                    st.write("-" * 10)
