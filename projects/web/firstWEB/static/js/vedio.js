var img_list;
var x = 0;
let Rimg = document.getElementById('Rimg');
let changeBt = document.getElementById('changeImg')
function subTime() {
    var form = new FormData(document.getElementById("ftime"));
    //使用
    function ajax() {
        console.log($('#ftime').serialize());
        $.ajax({
            url: "getRvedio",
            type: "POST",
            dataType: "json", //返回值的类型，不知道写成json对不对
            data: form,
            contentType: false, //这个字段啥意思?
            processData: false,
            success: function (res) { //得到图片的路径列表
                img_list = res;
                //console.log(res);
                getClick();
                alert("传输成功\n");
            },
            error: function (error) {
                alert("出错了,看看时间选对了吗?\n");
                console.log('error:', error);
            }
        });
    }
    ajax();
}



function changeImg() {
    var imgs = JSON.parse(img_list);
    console.log(x);
    try {

        // console.log(imgs[x].fields.img_url);
        // console.log(imgs[x].fields.img_id);
        Rimg.src = imgs[x].fields.img_url;
        x++;
    } catch (e) {
        //如果到了最后就不用播放了
        //console.log("结束或者出错");
        console.log(e);
        clearInterval(flag);
        x = 0; //回到视频的原点
    }
}

var flag;

function getClick() {
    changeBt.click();
}

//点击点击开始
function start() {
    flag = setInterval(getClick, 200);
}