import streamlit as st
import pandas as pd
import base64
import requests

def app():
    
    # st.header('So sánh')
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
    img_url = "https://i.pinimg.com/736x/92/c3/63/92c363695c754fca5924c79e41a3c6cb.jpg"  
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

        <div class="recommend-message">COMPARE</div>
        """
        st.sidebar.write(message_html, unsafe_allow_html=True)
        
    data = pd.read_excel("Recommend_cars/Data/data_Hang_Xe.xlsx")
    #data = pd.read_csv("data_Hang_Xe.csv")
    col1, col2 = st.columns(2)

    with col1:
        # Widget để chọn Hãng xe 1
        selected_brands_1 = st.selectbox("Chọn Hãng xe 1:", data['Hãng xe'].unique())
        if selected_brands_1:
            # Lọc Dòng xe dựa trên Hãng xe đã chọn
            filtered_vehicles_1 = data.loc[data['Hãng xe'] == selected_brands_1, 'Dòng xe'].unique()
            # Widget để chọn Dòng xe 1
            selected_vehicles_1 = st.selectbox("Chọn Dòng xe 1:", filtered_vehicles_1)
            # Hiển thị Phiên bản
            if selected_vehicles_1:
                filtered_versions_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1), 'Phiên bản'].unique()
                selected_versions_1 = st.selectbox("Chọn Phiên bản 1:", filtered_versions_1)
                st.write("CHỌN XE: ", str(selected_brands_1 + " " + selected_vehicles_1 + " " + selected_versions_1).upper())
                if 'Image_URL' in data.columns:
                    # Lấy Image_URL
                    Image_URL_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Image_URL'].values[0]
                    st.image(Image_URL_1)

    with col2:
        # Widget để chọn Hãng xe 2
        selected_brands_2 = st.selectbox("Chọn Hãng xe 2:", data['Hãng xe'].unique())
        if selected_brands_2:
            # Lọc Dòng xe dựa trên Hãng xe đã chọn
            filtered_vehicles_2 = data.loc[data['Hãng xe'] == selected_brands_2, 'Dòng xe'].unique()
            # Widget để chọn Dòng xe 2
            selected_vehicles_2 = st.selectbox("Chọn Dòng xe 2:", filtered_vehicles_2)
            # Hiển thị Phiên bản
            if selected_vehicles_2:
                filtered_versions_2 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2), 'Phiên bản'].unique()
                selected_versions_2 = st.selectbox("Chọn Phiên bản 2:", filtered_versions_2)
                st.write("CHỌN XE: ", str(selected_brands_2 + " " + selected_vehicles_2 + " " + selected_versions_2).upper())
                if 'Image_URL' in data.columns:
                    # Lấy Image_URL
                    Image_URL_2 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Image_URL'].values[0]
                    st.image(Image_URL_2)

    # So sánh
    compares_button = st.button("Compare")

    # Nếu nút lọc được nhấn
    if compares_button:
        col1, col2 = st.columns(2)
        with col1:
            st.write("| " + str(selected_brands_1 + " " + selected_vehicles_1 + " " + selected_versions_1))
            # Lấy giá tiền
            price_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Đơn giá'].values[0]
            st.caption(f" Giá bán: {price_1}")
            
        with col2:
            st.write("| " + str(selected_brands_2 + " " + selected_vehicles_2 + " " + selected_versions_2))
            # Lấy giá tiền
            price_2 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Đơn giá'].values[0]
            st.caption(f" Giá bán: {price_2}")
            
        col1, col2 = st.columns(2)
        with col1: 
            if 'Kích thước - Trọng lượng' in data.columns:
                # Lấy giá trị "Kích thước - Trọng lượng"
                size1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Kích thước - Trọng lượng'].values[0]
                # Kiểm tra kiểu dữ liệu
                # Loại bỏ dấu []
                size1 = size1.replace('[', '')
                size1 = size1.replace(']', '')

                # Loại bỏ dấu {}
                size1 = size1.replace('{', '')
                size1 = size1.replace('}', '')
                
                # Thêm {} bên ngoài
                size1 = "{{{}}}".format(size1)
                # Chuyển đổi thành dictionary
                size1_dict = eval(size1)
                st.caption("Kích thước - Trọng lượng")
                # Tạo DataFrame từ dictionary
                data_size1 = pd.DataFrame.from_dict(size1_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_size1)
            else:
                # Hiển thị thông báo
                st.caption("Kích thước - Trọng lượng")
        with col2:
            if 'Kích thước - Trọng lượng' in data.columns:
                # Lấy giá trị "Kích thước - Trọng lượng"
                size1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Kích thước - Trọng lượng'].values[0]
                # Loại bỏ dấu []
                size1 = size1.replace('[', '')
                size1 = size1.replace(']', '')

                # Loại bỏ dấu {}
                size1 = size1.replace('{', '')
                size1 = size1.replace('}', '')
                # Thêm {} bên ngoài
                size1 = "{{{}}}".format(size1)
                # Chuyển đổi thành dictionary
                size1_dict = eval(size1)
                st.caption("Kích thước - Trọng lượng")
                # Tạo DataFrame từ dictionary
                data_size1 = pd.DataFrame.from_dict(size1_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_size1)
            else:
                # Hiển thị thông báo
                st.caption("Kích thước - Trọng lượng")
            
        col1, col2 = st.columns(2)
        with col1:
            if 'Động cơ - Hộp số' in data.columns:
                # Lấy giá trị "Động cơ - Hộp số"
                dongco_hopso_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Động cơ - Hộp số'].values[0]
                if len(dongco_hopso_1) == 0:
                    st.caption("Động cơ - Hộp số")
                else:
                    # Chuyển đổi thành dictionary
                    # Loại bỏ dấu []
                    dongco_hopso_1 = dongco_hopso_1.replace('[', '')
                    dongco_hopso_1 = dongco_hopso_1.replace(']', '')

                    # Loại bỏ dấu {}
                    dongco_hopso_1 = dongco_hopso_1.replace('{', '')
                    dongco_hopso_1 = dongco_hopso_1.replace('}', '')
                    # Thêm {} bên ngoài
                    dongco_hopso_1 = "{{{}}}".format(dongco_hopso_1)
                    dongco_hopso_dict_1 = eval(dongco_hopso_1)
                # Hiển thị tiêu đề
                st.caption("Động cơ - Hộp số")
                # Tạo DataFrame từ dictionary
                data_dongco_hopso_1 = pd.DataFrame.from_dict(dongco_hopso_dict_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_dongco_hopso_1)
            else:
                # Hiển thị thông báo
                st.caption("Động cơ - Hộp số")
                
            if 'Động cơ - Mã lực' in data.columns:
                dongco_hopso_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Động cơ - Mã lực'].values[0]
                if not isinstance(dongco_hopso_1, (int, float, complex)):
                    if "{" not in dongco_hopso_1:
                        # Chuyển đổi sang từ điển
                        dongco_hopso_1 = {'Động cơ - Mã lực': dongco_hopso_1}
                    else:
                    # Chuyển đổi thành dictionary
                    # Loại bỏ dấu []
                        dongco_hopso_1 = dongco_hopso_1.replace('[', '')
                        dongco_hopso_1 = dongco_hopso_1.replace(']', '')

                        # Loại bỏ dấu {}
                        dongco_hopso_1 = dongco_hopso_1.replace('{', '')
                        dongco_hopso_1 = dongco_hopso_1.replace('}', '')
                        # Thêm {} bên ngoài
                        dongco_hopso_1 = "{{{}}}".format(dongco_hopso_1)
                        dongco_hopso_1 = eval(dongco_hopso_1)
                    # Hiển thị tiêu đề
                    st.caption("Động cơ - Mã lực")
                    # Tạo DataFrame từ dictionary
                    data_dongco_hopso_1 = pd.DataFrame.from_dict(dongco_hopso_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                    # Hiển thị bảng
                    st.dataframe(data_dongco_hopso_1)
                    
        with col2:
            if 'Động cơ - Hộp số' in data.columns:
                # Lấy giá trị "Động cơ - Hộp số"
                dongco_hopso_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Động cơ - Hộp số'].values[0]
                if not isinstance(dongco_hopso_1, (int, float, complex)):
                    # Chuyển đổi thành dictionary
                    # Loại bỏ dấu []
                    dongco_hopso_1 = dongco_hopso_1.replace('[', '')
                    dongco_hopso_1 = dongco_hopso_1.replace(']', '')

                    # Loại bỏ dấu {}
                    dongco_hopso_1 = dongco_hopso_1.replace('{', '')
                    dongco_hopso_1 = dongco_hopso_1.replace('}', '')
                    # Thêm {} bên ngoài
                    dongco_hopso_1 = "{{{}}}".format(dongco_hopso_1)
                    dongco_hopso_1 = eval(dongco_hopso_1)
                # Hiển thị tiêu đề
                st.caption("Động cơ - Hộp số")
                # Tạo DataFrame từ dictionary
                data_dongco_hopso_1 = pd.DataFrame.from_dict(dongco_hopso_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_dongco_hopso_1)
            else:
                # Hiển thị thông báo
                st.caption("Động cơ - Hộp số")

            if 'Động cơ - Mã lực' in data.columns:
                # Lấy giá trị "Động cơ - Hộp số"
                dongco_hopso_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Động cơ - Mã lực'].values[0]
                if not isinstance(dongco_hopso_1, (int, float, complex)):
                    if "{" not in dongco_hopso_1:
                        # Chuyển đổi sang từ điển
                        dongco_hopso_1 = {'Động cơ - Mã lực': dongco_hopso_1}
                    else:
                    # Chuyển đổi thành dictionary
                    # Loại bỏ dấu []
                        dongco_hopso_1 = dongco_hopso_1.replace('[', '')
                        dongco_hopso_1 = dongco_hopso_1.replace(']', '')

                        # Loại bỏ dấu {}
                        dongco_hopso_1 = dongco_hopso_1.replace('{', '')
                        dongco_hopso_1 = dongco_hopso_1.replace('}', '')
                        # Thêm {} bên ngoài
                        dongco_hopso_1 = "{{{}}}".format(dongco_hopso_1)
                        dongco_hopso_1 = eval(dongco_hopso_1)
                    # Hiển thị tiêu đề
                    st.caption("Động cơ - Mã lực")
                    # Tạo DataFrame từ dictionary
                    data_dongco_hopso_1 = pd.DataFrame.from_dict(dongco_hopso_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                    # Hiển thị bảng
                    st.dataframe(data_dongco_hopso_1)
                    
        col1, col2 = st.columns(2)
        with col1:
            # Lấy giá trị "Mức tiêu thụ nhiên liệu"
            muc_tieu_thu_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Mức tiêu thụ nhiên liệu'].values[0]                
            if not muc_tieu_thu_1:
                # Hiển thị thông báo
                st.caption("Mức tiêu thụ nhiên liệu")
            elif not isinstance(muc_tieu_thu_1, (int, float, complex)):
                # Chuyển đổi thành dictionary
                muc_tieu_thu_1 = eval(muc_tieu_thu_1)
                # Hiển thị tiêu đề
                st.caption("Mức tiêu thụ nhiên liệu")
                # Tạo DataFrame từ dictionary
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(muc_tieu_thu_1, orient='index', columns=['Giá trị thông số kĩ thuật']) 
                st.dataframe(data_muc_tieu_thu_1)               
            else:
                st.caption("Mức tiêu thụ nhiên liệu")

        with col2:
            # Lấy giá trị "Mức tiêu thụ nhiên liệu"
            muc_tieu_thu_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Mức tiêu thụ nhiên liệu'].values[0]                
            if not muc_tieu_thu_1:
                # Hiển thị thông báo
                st.caption("Mức tiêu thụ nhiên liệu")
            elif not isinstance(muc_tieu_thu_1, (int, float, complex)):
                # Chuyển đổi thành dictionary
                muc_tieu_thu_1 = eval(muc_tieu_thu_1)
                # Hiển thị tiêu đề
                st.caption("Mức tiêu thụ nhiên liệu")
                # Tạo DataFrame từ dictionary
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(muc_tieu_thu_1, orient='index', columns=['Giá trị thông số kĩ thuật']) 
                st.dataframe(data_muc_tieu_thu_1)               
            else:
                st.caption("Mức tiêu thụ nhiên liệu")
        
        col1, col2 = st.columns(2)
        with col1:
            he_thong_treo_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Truyền động - Hệ thống treo'].values[0]
            if not he_thong_treo_1:
                # Hiển thị thông báo
                st.caption("Truyền động - Hệ thống treo")
            elif isinstance(he_thong_treo_1, (int, float, complex)):
                # Chuyển đổi sang từ điển
                he_thong_treo_1 = {'Giá trị thông số kĩ thuật': he_thong_treo_1}
                # Hiển thị tiêu đề
                st.caption("Truyền động - Hệ thống treo")
                # Tạo DataFrame từ từ điển
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(he_thong_treo_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
            else:
                # Loại bỏ dấu []
                he_thong_treo_1 = he_thong_treo_1.replace('[', '')
                he_thong_treo_1 = he_thong_treo_1.replace(']', '')

                # Loại bỏ dấu {}
                he_thong_treo_1 = he_thong_treo_1.replace('{', '')
                he_thong_treo_1 = he_thong_treo_1.replace('}', '')
                # Thêm {} bên ngoài
                he_thong_treo_1 = "{{{}}}".format(he_thong_treo_1)
                # Chuyển đổi thành dictionary
                he_thong_treo_dict_1 = eval(he_thong_treo_1)
                # Hiển thị tiêu đề
                st.caption("Truyền động - Hệ thống treo")
                # Tạo DataFrame từ dictionary
                data_he_thong_treo_dict_1 = pd.DataFrame.from_dict(he_thong_treo_dict_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_he_thong_treo_dict_1)
        with col2:
                # Lấy giá trị "Mức tiêu thụ nhiên liệu"
            he_thong_treo_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Truyền động - Hệ thống treo'].values[0]
            if not he_thong_treo_1:
                # Hiển thị thông báo
                st.caption("Truyền động - Hệ thống treo")
            elif isinstance(he_thong_treo_1, (int, float, complex)):
                # Chuyển đổi sang từ điển
                he_thong_treo_1 = {'Giá trị thông số kĩ thuật': he_thong_treo_1}
                # Hiển thị tiêu đề
                st.caption("Truyền động - Hệ thống treo")
                # Tạo DataFrame từ từ điển
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(he_thong_treo_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
            else:
                # Loại bỏ dấu []
                he_thong_treo_1 = he_thong_treo_1.replace('[', '')
                he_thong_treo_1 = he_thong_treo_1.replace(']', '')

                # Loại bỏ dấu {}
                he_thong_treo_1 = he_thong_treo_1.replace('{', '')
                he_thong_treo_1 = he_thong_treo_1.replace('}', '')
                # Thêm {} bên ngoài
                he_thong_treo_1 = "{{{}}}".format(he_thong_treo_1)
                # Chuyển đổi thành dictionary
                he_thong_treo_dict_1 = eval(he_thong_treo_1)
                # Hiển thị tiêu đề
                st.caption("Truyền động - Hệ thống treo")
                # Tạo DataFrame từ dictionary
                data_he_thong_treo_dict_1 = pd.DataFrame.from_dict(he_thong_treo_dict_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_he_thong_treo_dict_1)
            
        col1, col2 = st.columns(2)
        with col1:
            khung_gam_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Khung gầm'].values[0]
            if not khung_gam_1:
                # Hiển thị thông báo
                st.caption("Khung gầm")
            elif isinstance(khung_gam_1, (int, float, complex)):
                # Chuyển đổi sang từ điển
                khung_gam_1 = {'Giá trị thông số kĩ thuật': khung_gam_1}
                # Hiển thị tiêu đề
                st.caption("Khung gầm")
                # Tạo DataFrame từ từ điển
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(khung_gam_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
            else:
                # Loại bỏ dấu []
                khung_gam_1 = khung_gam_1.replace('[', '')
                khung_gam_1 = khung_gam_1.replace(']', '')

                # Loại bỏ dấu {}
                khung_gam_1 = khung_gam_1.replace('{', '')
                khung_gam_1 = khung_gam_1.replace('}', '')
                # Thêm {} bên ngoài
                khung_gam_1 = "{{{}}}".format(khung_gam_1)
                # Chuyển đổi thành dictionary
                khung_gam_dict_1 = eval(khung_gam_1)
                # Hiển thị tiêu đề
                st.caption("Khung gầm")
                # Tạo DataFrame từ dictionary
                data_he_thong_treo_dict_1 = pd.DataFrame.from_dict(khung_gam_dict_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_he_thong_treo_dict_1)
        with col2:
            khung_gam_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Khung gầm'].values[0]
            if not khung_gam_1:
                # Hiển thị thông báo
                st.caption("Khung gầm")
            elif isinstance(khung_gam_1, (int, float, complex)):
                # Chuyển đổi sang từ điển
                khung_gam_1 = {'Giá trị thông số kĩ thuật': khung_gam_1}
                # Hiển thị tiêu đề
                st.caption("Khung gầm")
                # Tạo DataFrame từ từ điển
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(khung_gam_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
            else:
                # Loại bỏ dấu []
                khung_gam_1 = khung_gam_1.replace('[', '')
                khung_gam_1 = khung_gam_1.replace(']', '')

                # Loại bỏ dấu {}
                khung_gam_1 = khung_gam_1.replace('{', '')
                khung_gam_1 = khung_gam_1.replace('}', '')
                # Thêm {} bên ngoài
                khung_gam_1 = "{{{}}}".format(khung_gam_1)
                # Chuyển đổi thành dictionary
                khung_gam_dict_1 = eval(khung_gam_1)
                # Hiển thị tiêu đề
                st.caption("Khung gầm")
                # Tạo DataFrame từ dictionary
                data_he_thong_treo_dict_1 = pd.DataFrame.from_dict(khung_gam_dict_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_he_thong_treo_dict_1)
        
        col1, col2 = st.columns(2)
        with col1:
            khung_xe_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Khung xe'].values[0]
            if not khung_xe_1:
                # Hiển thị thông báo
                st.caption("Khung xe")
            elif isinstance(khung_xe_1, (int, float, complex)):
                # Chuyển đổi sang từ điển
                khung_xe_1 = {'Giá trị thông số kĩ thuật': khung_xe_1}
                # Hiển thị tiêu đề
                st.caption("Khung xe")
                # Tạo DataFrame từ từ điển
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(khung_xe_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
            else:
                # Loại bỏ dấu []
                khung_xe_1 = khung_xe_1.replace('[', '')
                khung_xe_1 = khung_xe_1.replace(']', '')

                # Loại bỏ dấu {}
                khung_xe_1 = khung_xe_1.replace('{', '')
                khung_xe_1 = khung_xe_1.replace('}', '')
                # Thêm {} bên ngoài
                khung_xe_1 = "{{{}}}".format(khung_xe_1)
                # Chuyển đổi thành dictionary
                khung_xe_1 = eval(khung_xe_1)
                # Hiển thị tiêu đề
                st.caption("Khung xe")
                # Tạo DataFrame từ dictionary
                data_dict_1 = pd.DataFrame.from_dict(khung_xe_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_dict_1)
        with col2:
            khung_xe_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Khung xe'].values[0]
            if not khung_xe_1:
                # Hiển thị thông báo
                st.caption("Khung xe")
            elif isinstance(khung_xe_1, (int, float, complex)):
                # Chuyển đổi sang từ điển
                khung_xe_1 = {'Giá trị thông số kĩ thuật': khung_xe_1}
                # Hiển thị tiêu đề
                st.caption("Khung xe")
                # Tạo DataFrame từ từ điển
                data_muc_tieu_thu_1 = pd.DataFrame.from_dict(khung_xe_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
            else:
                # Loại bỏ dấu []
                khung_xe_1 = khung_xe_1.replace('[', '')
                khung_xe_1 = khung_xe_1.replace(']', '')

                # Loại bỏ dấu {}
                khung_xe_1 = khung_xe_1.replace('{', '')
                khung_xe_1 = khung_xe_1.replace('}', '')
                # Thêm {} bên ngoài
                khung_xe_1 = "{{{}}}".format(khung_xe_1)
                # Chuyển đổi thành dictionary
                khung_xe_1 = eval(khung_xe_1)
                # Hiển thị tiêu đề
                st.caption("Khung xe")
                # Tạo DataFrame từ dictionary
                data_dict_1 = pd.DataFrame.from_dict(khung_xe_1, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_dict_1)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Ngoại thất' in data.columns:
                # Lấy giá trị "Ngoại thất"
                ngoai_that_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Ngoại thất'].values[0]
                # Loại bỏ dấu []
                ngoai_that_1 = ngoai_that_1.replace('[', '')
                ngoai_that_1 = ngoai_that_1.replace(']', '')
                # Loại bỏ dấu {}
                ngoai_that_1 = ngoai_that_1.replace('{', '')
                ngoai_that_1 = ngoai_that_1.replace('}', '')
                # Thêm {} bên ngoài
                ngoai_that_1 = "{{{}}}".format(ngoai_that_1)
                # Chuyển đổi thành dictionary
                ngoai_that_dict_1 = eval(ngoai_that_1)
                # Lọc các giá trị "-"
                ngoai_that_filtered_dict = {key: value for key, value in ngoai_that_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("Ngoại thất")
                # Tạo DataFrame từ dictionary
                data_ngoai_that_1 = pd.DataFrame.from_dict(ngoai_that_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_ngoai_that_1)
            else:
                # Hiển thị thông báo
                st.caption("Ngoại thất")
        with col2:
            if 'Ngoại thất' in data.columns:
                # Lấy giá trị "Ngoại thất"
                ngoai_that_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Ngoại thất'].values[0]
                # Loại bỏ dấu []
                ngoai_that_1 = ngoai_that_1.replace('[', '')
                ngoai_that_1 = ngoai_that_1.replace(']', '')
                # Loại bỏ dấu {}
                ngoai_that_1 = ngoai_that_1.replace('{', '')
                ngoai_that_1 = ngoai_that_1.replace('}', '')
                # Thêm {} bên ngoài
                ngoai_that_1 = "{{{}}}".format(ngoai_that_1)
                # Chuyển đổi thành dictionary
                ngoai_that_dict_1 = eval(ngoai_that_1)
                # Lọc các giá trị "-"
                ngoai_that_filtered_dict = {key: value for key, value in ngoai_that_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("Ngoại thất")
                # Tạo DataFrame từ dictionary
                data_ngoai_that_1 = pd.DataFrame.from_dict(ngoai_that_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_ngoai_that_1)
            else:
                # Hiển thị thông báo
                st.caption("Ngoại thất")
            
        col1, col2 = st.columns(2)
        with col1:
            if 'Nội thất' in data.columns:
                # Lấy giá trị "Nội thất"
                noi_that_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Nội thất'].values[0]
                # Loại bỏ dấu []
                noi_that_1 = noi_that_1.replace('[', '')
                noi_that_1 = noi_that_1.replace(']', '')
                # Loại bỏ dấu {}
                noi_that_1 = noi_that_1.replace('{', '')
                noi_that_1 = noi_that_1.replace('}', '')
                # Thêm {} bên ngoài
                noi_that_1 = "{{{}}}".format(noi_that_1)
                # Chuyển đổi thành dictionary
                noi_that_dict_1 = eval(noi_that_1)
                # Lọc các giá trị "-"
                noi_that_filtered_dict = {key: value for key, value in noi_that_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("Nội thất")
                # Tạo DataFrame từ dictionary
                data_noi_that_1 = pd.DataFrame.from_dict(noi_that_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_noi_that_1)
            else:
                # Hiển thị thông báo
                st.caption("Nội thất")
        with col2:
            if 'Nội thất' in data.columns:
                # Lấy giá trị "Nội thất"
                noi_that_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Nội thất'].values[0]
                # Loại bỏ dấu []
                noi_that_1 = noi_that_1.replace('[', '')
                noi_that_1 = noi_that_1.replace(']', '')
                # Loại bỏ dấu {}
                noi_that_1 = noi_that_1.replace('{', '')
                noi_that_1 = noi_that_1.replace('}', '')
                # Thêm {} bên ngoài
                noi_that_1 = "{{{}}}".format(noi_that_1)
                # Chuyển đổi thành dictionary
                noi_that_dict_1 = eval(noi_that_1)
                # Lọc các giá trị "-"
                noi_that_filtered_dict = {key: value for key, value in noi_that_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("Nội thất")
                # Tạo DataFrame từ dictionary
                data_noi_that_1 = pd.DataFrame.from_dict(noi_that_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_noi_that_1)
            else:
                # Hiển thị thông báo
                st.caption("Nội thất")
        
        col1, col2 = st.columns(2)
        with col1:
            ghe = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Ghế'].values[0]
            if not ghe:
                st.caption("Ghế")
            elif isinstance(ghe, (int, float, complex)):
                st.caption("Ghế")
            else:
                # Loại bỏ dấu []
                ghe = ghe.replace('[', '')
                ghe = ghe.replace(']', '')

                # Loại bỏ dấu {}
                ghe = ghe.replace('{', '')
                ghe = ghe.replace('}', '')
                # Thêm {} bên ngoài
                ghe = "{{{}}}".format(ghe)
                # Chuyển đổi thành dictionary
                ghe = eval(ghe)
                # Hiển thị tiêu đề
                st.caption("Ghế")
                # Tạo DataFrame từ dictionary
                data_ghe = pd.DataFrame.from_dict(ghe, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_ghe)
        with col2:
            ghe = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Ghế'].values[0]
            if not ghe:
                st.caption("Ghế")
            elif isinstance(ghe, (int, float, complex)):
                st.caption("Ghế")
            else:
                # Loại bỏ dấu []
                ghe = ghe.replace('[', '')
                ghe = ghe.replace(']', '')

                # Loại bỏ dấu {}
                ghe = ghe.replace('{', '')
                ghe = ghe.replace('}', '')
                # Thêm {} bên ngoài
                ghe = "{{{}}}".format(ghe)
                # Chuyển đổi thành dictionary
                ghe = eval(ghe)
                # Hiển thị tiêu đề
                st.caption("Ghế")
                # Tạo DataFrame từ dictionary
                data_ghe = pd.DataFrame.from_dict(ghe, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_ghe)
                
        col1, col2 = st.columns(2)
        with col1:
            # Lấy giá trị "Nội thất"
            tien_nghi_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'Tiện nghi'].values[0]
            if not tien_nghi_1:
                st.caption("Tiện nghi")
            elif isinstance(tien_nghi_1, (int, float, complex)):
                st.caption("Tiện nghi")
            else:
                # Loại bỏ dấu []
                tien_nghi_1 = tien_nghi_1.replace('[', '')
                tien_nghi_1 = tien_nghi_1.replace(']', '')
                # Loại bỏ dấu {}
                tien_nghi_1 = tien_nghi_1.replace('{', '')
                tien_nghi_1 = tien_nghi_1.replace('}', '')
                # Thêm {} bên ngoài
                tien_nghi_1 = "{{{}}}".format(tien_nghi_1)
                # Chuyển đổi thành dictionary
                tien_nghi_dict_1 = eval(tien_nghi_1)
                # Lọc các giá trị "-"
                noi_that_filtered_dict = {key: value for key, value in tien_nghi_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("Tiện nghi")
                # Tạo DataFrame từ dictionary
                data_tien_nghi_1 = pd.DataFrame.from_dict(noi_that_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_tien_nghi_1)
        with col2:
            # Lấy giá trị "Nội thất"
            tien_nghi_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'Tiện nghi'].values[0]
            if not tien_nghi_1:
                st.caption("Tiện nghi")
            elif isinstance(tien_nghi_1, (int, float, complex)):
                st.caption("Tiện nghi")
            else:
                # Loại bỏ dấu []
                tien_nghi_1 = tien_nghi_1.replace('[', '')
                tien_nghi_1 = tien_nghi_1.replace(']', '')
                # Loại bỏ dấu {}
                tien_nghi_1 = tien_nghi_1.replace('{', '')
                tien_nghi_1 = tien_nghi_1.replace('}', '')
                # Thêm {} bên ngoài
                tien_nghi_1 = "{{{}}}".format(tien_nghi_1)
                # Chuyển đổi thành dictionary
                tien_nghi_dict_1 = eval(tien_nghi_1)
                # Lọc các giá trị "-"
                noi_that_filtered_dict = {key: value for key, value in tien_nghi_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("Tiện nghi")
                # Tạo DataFrame từ dictionary
                data_tien_nghi_1 = pd.DataFrame.from_dict(noi_that_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_tien_nghi_1)
        
        col1, col2 = st.columns(2)
        with col1:
                # Lấy giá trị "An toàn"
            an_toan_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'An toàn'].values[0]
            if not an_toan_1:
                st.caption("An toàn")
            elif isinstance(an_toan_1, (int, float, complex)):
                st.caption("An toàn")
            else:
                # Loại bỏ dấu []
                an_toan_1 = an_toan_1.replace('[', '')
                an_toan_1 = an_toan_1.replace(']', '')
                # Loại bỏ dấu {}
                an_toan_1 = an_toan_1.replace('{', '')
                an_toan_1 = an_toan_1.replace('}', '')
                # Thêm {} bên ngoài
                an_toan_1 = "{{{}}}".format(an_toan_1)
                # Chuyển đổi thành dictionary
                an_toan_dict_1 = eval(an_toan_1)
                # Lọc các giá trị "-"
                an_toan_filtered_dict = {key: value for key, value in an_toan_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("An toàn")
                # Tạo DataFrame từ dictionary
                data_an_toan_1 = pd.DataFrame.from_dict(an_toan_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_an_toan_1)
        with col2:
            an_toan_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'An toàn'].values[0]
            if not an_toan_1:
                st.caption("An toàn")
            elif isinstance(an_toan_1, (int, float, complex)):
                st.caption("An toàn")
            else:
                # Loại bỏ dấu []
                an_toan_1 = an_toan_1.replace('[', '')
                an_toan_1 = an_toan_1.replace(']', '')
                # Loại bỏ dấu {}
                an_toan_1 = an_toan_1.replace('{', '')
                an_toan_1 = an_toan_1.replace('}', '')
                # Thêm {} bên ngoài
                an_toan_1 = "{{{}}}".format(an_toan_1)
                # Chuyển đổi thành dictionary
                an_toan_dict_1 = eval(an_toan_1)
                # Lọc các giá trị "-"
                an_toan_filtered_dict = {key: value for key, value in an_toan_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("An toàn")
                # Tạo DataFrame từ dictionary
                data_an_toan_1 = pd.DataFrame.from_dict(an_toan_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_an_toan_1)
            
        col1, col2 = st.columns(2)
        with col1:
            an_ninh_1 = data.loc[(data['Hãng xe'] == selected_brands_1) & (data['Dòng xe'] == selected_vehicles_1) & (data['Phiên bản'] == selected_versions_1), 'An ninh'].values[0]
            if not an_ninh_1:
                st.caption("An ninh")
            elif isinstance(an_ninh_1, (int, float, complex)):
                st.caption("An ninh")
            else:
                # Loại bỏ dấu []
                an_ninh_1 = an_ninh_1.replace('[', '')
                an_ninh_1 = an_ninh_1.replace(']', '')
                # Loại bỏ dấu {}
                an_ninh_1 = an_ninh_1.replace('{', '')
                an_ninh_1 = an_ninh_1.replace('}', '')
                # Thêm {} bên ngoài
                an_ninh_1 = "{{{}}}".format(an_ninh_1)
                # Chuyển đổi thành dictionary
                an_ninh_dict_1 = eval(an_ninh_1)
                # Lọc các giá trị "-"
                an_ninh_filtered_dict = {key: value for key, value in an_ninh_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("An ninh")
                # Tạo DataFrame từ dictionary
                data_an_ninh_1 = pd.DataFrame.from_dict(an_ninh_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_an_ninh_1)
                
        with col2:
            an_ninh_1 = data.loc[(data['Hãng xe'] == selected_brands_2) & (data['Dòng xe'] == selected_vehicles_2) & (data['Phiên bản'] == selected_versions_2), 'An ninh'].values[0]
            if not an_ninh_1:
                st.caption("An ninh")
            elif isinstance(an_ninh_1, (int, float, complex)):
                st.caption("An ninh")
            else:
                # Loại bỏ dấu []
                an_ninh_1 = an_ninh_1.replace('[', '')
                an_ninh_1 = an_ninh_1.replace(']', '')
                # Loại bỏ dấu {}
                an_ninh_1 = an_ninh_1.replace('{', '')
                an_ninh_1 = an_ninh_1.replace('}', '')
                # Thêm {} bên ngoài
                an_ninh_1 = "{{{}}}".format(an_ninh_1)
                # Chuyển đổi thành dictionary
                an_ninh_dict_1 = eval(an_ninh_1)
                # Lọc các giá trị "-"
                an_ninh_filtered_dict = {key: value for key, value in an_ninh_dict_1.items() if value != "-" and value != ""}
                # Hiển thị tiêu đề
                st.caption("An ninh")
                # Tạo DataFrame từ dictionary
                data_an_ninh_1 = pd.DataFrame.from_dict(an_ninh_filtered_dict, orient='index', columns=['Giá trị thông số kĩ thuật'])
                # Hiển thị bảng
                st.dataframe(data_an_ninh_1)
        
        st.write("-" * 10)
