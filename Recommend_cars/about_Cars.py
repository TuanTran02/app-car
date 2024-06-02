import streamlit as st
import base64
import requests

def app():
    
    # st.header('Giới thiệu')
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
    img_url = "https://i.pinimg.com/originals/bc/62/61/bc6261dcd0ec609d86121d9f02390bf2.jpg"  # Replace with your desired image URL
    sidebar_bg(img_url)
    with st.sidebar:
        message_html = """
        <style>
            .recommend-message {
                color: white; 
                font-weight: bold;
                font-size: 45px;
            }
        </style>

        <div class="recommend-message">About !</div>
        """
        st.sidebar.write(message_html, unsafe_allow_html=True)
        
    # st.markdown("<h1 style='font-size: 24px;'>Giới thiệu</h1>", unsafe_allow_html=True)
    st.markdown("""
Bạn đang tìm kiếm một chiếc xe hơi phù hợp với nhu cầu và ngân sách của mình? Vậy thì bạn đã đến đúng nơi rồi! Trang web đề xuất xe hơi của chúng tôi sẽ giúp bạn dễ dàng tìm thấy chiếc xe mơ ước một cách nhanh chóng và hiệu quả.

Tại đây, bạn có thể:

- Lọc tìm kiếm theo các tiêu chí khác nhau như hãng xe, kiểu xe, giá cả, mức tiêu hao nhiên liệu, tính năng, ...
- Dự đoán giá xe theo các tiêu chí khác nhau.
- So sánh các mẫu xe cạnh tranh trực tiếp về giá cả, thông số kỹ thuật và tính năng.
- Tìm kiếm các ưu đãi và khuyến mãi cho xe mới và xe đã qua sử dụng.
- Liên hệ trực tiếp với các đại lý xe hơi để được tư vấn và báo giá cụ thể.
Với đội ngũ chuyên gia giàu kinh nghiệm và kho dữ liệu khổng lồ về xe hơi, chúng tôi cam kết mang đến cho bạn những đề xuất xe chính xác và phù hợp nhất với nhu cầu của bạn.

Hãy truy cập trang web của chúng tôi ngay hôm nay để bắt đầu hành trình tìm kiếm chiếc xe hoàn hảo của bạn!

Với trang web đề xuất xe hơi của chúng tôi, việc mua xe hơi sẽ trở nên dễ dàng và thú vị hơn bao giờ hết!
""")
    st.markdown('Created by: [Trần Ngọc Tuấn](https://github.com/TuanTran02)')
    st.markdown('Created by: [Huỳnh Nhật Linh](https://github.com/nhatlinh1312/Recommend_RPC_Cars)')