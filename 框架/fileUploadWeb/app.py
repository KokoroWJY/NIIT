# 导入所有依赖包
import flask
from werkzeug.utils import secure_filename
import os

# 创建一个flask web应用，名称为fileUploadWeb，实现任意类型文件的上传和保存
app = flask.Flask("fileUploadWeb")


def upload_file():
    print("开始处理文件上传...")
    if flask.request.method == "POST":  # 设置request的模式为POST
        print("进入upload_file()处理流程...")
        file_upload = flask.request.files["upload_file"]  # 获取文件
        print("原始文件名：" + file_upload.filename)
        secure_filename = file_upload.filename
        print("转换成没有乱码的文件名：" + secure_filename)
        file_path = os.path.join(app.root_path, "save_file/" + secure_filename)  # 获取文件的保存路径
        print("保存文件的文件路径：" + file_path)
        file_upload.save(file_path)  # 将文件保存在应用的根目录下
        print("文件上传成功.")

        return flask.render_template(template_name_or_list="result.html")
    return "文件上传失败."


# 增加upload路由，使用POST方法，用于文件的上传
app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_file,
                 methods=["POST"])  # 绑定视图函数，将函数upload_file()与网址"/upload/"绑定起来


def redirect_to_upload():
    return flask.render_template(template_name_or_list="upload_file.html")  # 将网页重定向到upload_file.html页面


app.add_url_rule(rule="/", endpoint="homepage", view_func=redirect_to_upload)  # 将主页地址"/"与视图函数redirect_to_upload()绑定起来

if __name__ == "__main__":
    app.run(debug=False)  # 系统默认为：host="127.0.0.1", port=5000
