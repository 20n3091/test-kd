# 写进「app.py」
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("recognition app")
st.sidebar.write("Determine what the image is using the original image recognition model。")

st.sidebar.write("")

img_source = st.sidebar.radio("
Select image source",                             ("upload image", "taken with a camera"))
if img_source == "upload image":
    img_file = st.sidebar.file_uploader("Please select an image。", type=["png", "jpg"])
elif img_source == "taken with a camera":
    img_file = st.camera_input("taken with a camera")

if img_file is not None:
    with st.spinner("Estimating..."):
        img = Image.open(img_file)
        st.image(img, caption="target image", width=480)
        st.write("")

        # 预测
        results = predict(img)

        # 查看结果
        st.subheader("查看结果")
        n_top = 3  
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%的可能性是" + result[0] )

        # 饼图展示
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 饼图
        st.pyplot(fig)
