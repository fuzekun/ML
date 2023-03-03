
    //访问用户媒体设备的兼容方法
    function getUserMedia(constraints, success, error) {
        if (navigator.mediaDevices.getUserMedia) {
            //最新的标准API
            navigator.mediaDevices.getUserMedia(constraints).then(success).catch(error);
        } else if (navigator.webkitGetUserMedia) {
            //webkit核心浏览器
            navigator.webkitGetUserMedia(constraints, success, error)
        } else if (navigator.mozGetUserMedia) {
            //firfox浏览器
            navigator.mozGetUserMedia(constraints, success, error);
        } else if (navigator.getUserMedia) {
            //旧版API
            navigator.getUserMedia(constraints, success, error);
        }
    }

    let capture = document.getElementById('capture');
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let context = canvas.getContext('2d');

    function success(stream){
         //兼容webkit核心浏览器
        let CompatibleURL = window.URL || window.webkitURL;
        //将视频流设置为video元素的源
        console.log(stream);

        //video.src = CompatibleURL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
    }
    function  error(error){
        console.log(`访问用户媒体设备失败${error.name}, ${error.message}`);
    }

    //这一段用来展示相应的
    if (navigator.mediaDevices.getUserMedia || navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
        //调用用户媒体设备, 访问摄像头
        getUserMedia({video: {width: 480, height: 320}}, success, error);
    } else {
        alert('不支持访问用户媒体');
    }

    //点击拍照按钮
    capture.addEventListener('click',function (){
        context.drawImage(video,0, 0, 480, 320);


        //200毫秒后将图片传给后台
        setTimeout(() => {
            sent()
        }, 500)
    })

     //两个辅助函数，将图片转化为相应的文件，并且进行编码
     function dataURItoBlob(base64Data) {
        var byteString;
        if (base64Data.split(',')[0].indexOf('base64') >= 0)
            byteString = atob(base64Data.split(',')[1]);
        else
            byteString = unescape(base64Data.split(',')[1]);
        var mimeString = base64Data.split(',')[0].split(':')[1].split(';')[0];
        var ia = new Uint8Array(byteString.length);
        for (var i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ia], {type: mimeString});
    };

    //将blob转换为file
    function blobToFile(theBlob, fileName) {
        theBlob.lastModifiedDate = new Date();
        theBlob.name = fileName;
        return theBlob;
    };

    //传输函数
    function sent(){
          //生成图片格式base64包括：jpg、png格式
        var base64Data = canvas.toDataURL("image/jpeg", 1.0);
        //封装blob对象
        var blob = dataURItoBlob(base64Data);

        var file = blobToFile(blob, "imgSignature");

        //这里我打算用ajax方式，所以需要使用Formdata形式
        var data = new FormData();
        data.append("file", file);

        function ajax() {
             $.ajax({
                //想一想对应的相对路径应该怎么写？
                url: "http://localhost:8000/firstWEB/getFireImg",
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
        capture.click();
    }
    setInterval(getClick, 200);