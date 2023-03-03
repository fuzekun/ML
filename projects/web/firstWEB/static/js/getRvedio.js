
    var x = 109;
    let capture = document.getElementById('capture');

    //点击拍照按钮
    capture.addEventListener('click',function (){


        //200毫秒后将id传递给后台
        setTimeout(() => {
            sent()
        }, 200)
    })
    //传输函数
    function sent(){
        var data = new FormData();
        data.append("id", x);
        console.log(x);
        function ajax() {
             $.ajax({
                //想一想对应的相对路径应该怎么写？
                url: "http://localhost:8000/firstWEB/getRvedio",
                type : "POST",
                dataType: "text",
                data: data,
                contentType: false, //这个字段啥意思?
                processData: false,

                 //这里进行处理相应的危险信息以及相应的图片展示
                 //可以将图片保存在文件中，之后前端进行获取图片就行了
                success :function (res) {
                    //将图片的BASE64编码设置给src
                    //alert("请求成功");
                    //console.log(res);
                     $("#imgP").attr("src","data:image/jpg;base64,"+res);

                    //context.dfrawImage(video,0, 0, 480, 320);
                },
                 error :function (error){
                    //alert("请求失败");
                    console.log('error', error);
                 }
            });
        }
        ajax();
    }
    <!--自动获取图像的函数-->
    function getClick() {
        x++;
        capture.click();
    }
    setInterval(getClick, 200);