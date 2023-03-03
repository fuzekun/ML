function $(id) {//$是一个函数名.要是页面中和jq出现冲突。可以改成任意函数名$("date")调用这个函数
    return document.getElementById(id);
}

function change(cls) {
    return document.getElementsByClassName(cls);
}

//获取日期
var day1;

function getDateNum(e) {
    console.log(e && e.target.nodeName)//打印发生事件的元素，再获取元素nodeName
    if (e.target.nodeName == "LI") {//先判断nodeName是LI
        if (e.target.className != "gray") {//点击本月的日期，显示在日期
            day1 = e.target.innerHTML
            change("content")[0].innerHTML = year + '年' + (month + 1) + '月' + day1 + '日';
        } else {//点击灰色日期即（上个月或者下个月日期）切换到当月
            if (e.target.innerHTML > 14) {
                lastMonth();
            } else {
                nextMonth();
            }
        }
    }
}

//获取年、月
var today = new Date();
var year = today.getFullYear(), month = today.getMonth();
var totalDay;
var uls = $("date").children, list;

function loadCalendar() {
    totalDay = getMonthDays(year, month + 1);//计算一个月的天数
    var firstDay = (new Date(year, month, 1)).getDay();//计算每个月1号在星期几
    var lastMonthDay = getMonthDays(year, month);
    var lastDayCal = lastMonthDay - firstDay + 1;//计算上个月在第一行显示的天数
    //获取ul元素
    var num = 1, nextNum = 1;//日期显示
    // 类数组对象 转 数组
    //uls = Array.prototype.slice.call(uls)
    //获取li元素 填充
    for (var r = 0; r < uls.length; r++) {
        list = uls[r].children;//遍历ul,获得li
        for (var line = 0; line < list.length; line++) {
            if (r == 0) {//在第一行 与第一天进行判断 大于等于第一天时载入日期
                if (line >= firstDay) {
                    list[line].innerHTML = num++;
                    list[line].setAttribute("class", "");
                } else {
                    list[line].innerHTML = lastDayCal++;//第一行的上个月天数显示
                    list[line].setAttribute("class", "gray");
                }
            } else {
                //判断是否超出天数 ,不超出则继续加，超出则显示下个月日期
                if (num <= totalDay) {
                    list[line].setAttribute("class", "");
                    list[line].innerHTML = num++;
                } else {
                    list[line].innerHTML = nextNum++;//下个月日期显示
                    list[line].setAttribute("class", "gray");
                }
            }
        }
    }
}

loadCalendar();

function getMonthDays(year, month) {
    //判断2月份天数
    if (month == 2) {
        days = (year % 4 == 0) && (year % 100 != 0) || (year % 400 == 0) ? 29 : 28;
    } else {
        //1-7月 单数月为31日
        if (month < 7) {
            days = month % 2 == 1 ? 31 : 30;
        } else {
            //8-12月 双月为31日
            days = month % 2 == 0 ? 31 : 30;
        }
    }
    return days;
}

//右击箭头下个月
//change("rf-tri")[0].onclick =
function nextMonth() {
    month++;
    if (month > 11) {
        year += 1;
        month = 0;
    }
    change("content")[0].innerHTML = year + "年" + (month + 1) + "月" + "1日";
    //console.log(month+1);
    loadCalendar();
}

//左击箭头上个月
//change("lf-tri")[0].onclick =
function lastMonth() {
    month--;
    if (month < 0) {
        month = 11;
        year -= 1;
    }
    change("content")[0].innerHTML = year + "年" + (month + 1) + "月" + "1日";
    //console.log(month+1);
    loadCalendar();
}

//测试变量是否为所需
function print() {
    var mymonth = month + 1;
    alert("年:" + year);
    alert("月:" + mymonth);
    alert("日:" + day1);
}

//传递年月日，同时获取相应的图片
var img_list;

function sent() {
    //注意不要改变全局变量，另外,month进行格式化
    var mymonth = month + 1;
    if (mymonth < 10) {
        mymonth = "0" + mymonth;
    }
    var time = year + "-" + mymonth + "-" + day1;
    var data = new FormData();
    data.append("time", time);

    function ajax() {
        $.ajax({
            url: "getRvedio",
            type: "POST",
            dataType: "json", //返回值的类型，不知道写成json对不对
            data: data,
            contentType: false, //这个字段啥意思?
            processData: false,
            success: function (res) { //得到图片的路径列表
                img_list = res;
                //console.log(res);
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


var x = 0;
let Rimg = document.getElementById('Rimg');
let changeBt = document.getElementById('changeImg')

function changeImg() {
    var imgs = JSON.parse(img_list);
    console.log(x);
    try {

        console.log(imgs[x].fields.img_url);
        console.log(imgs[x].fields.img_id);
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