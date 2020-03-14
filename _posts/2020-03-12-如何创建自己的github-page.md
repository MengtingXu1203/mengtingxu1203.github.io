---
title: 如何用最简单的方法搭建自己的github-page
date:  2020-03-14 13:59:13 +0800
category: 杂记
tags: 博客
excerpt: 介绍如何搭建github-page
---

博客最主要的目的是记录自己生活/工作/技术上的点点滴滴，其次才是和大家分享。

### 1. 写在前面
#### 1.1. 如何通过ssh协议连接远程仓库github

<font color = 'red'>首先确保自己的电脑已经正常安装git和ssh服务</font>

##### a. 首先设置username和email.
注意："XXX"中的内容请改成自己的github用户名和github注册的邮箱
>git config - -global user.name "selous123"

>git config - -global user.email "lrhselous@163.com"

##### b. 通过终端命令生成ssh keys， 全部回车设置为默认值
>$ ssh-keygen -t rsa -C "lrhselous@163.com"

>Generating public/private rsa key pair.

>Enter file in which to save the key (/c/Users/selous/.ssh/id_rsa):

##### c. 复制ssh公钥信息
如果上述ssh命令成功，则默认在主目录下(如上述命令输出：Enter file in which to save the key (**/c/Users/selous/.ssh**/id_rsa))， 就可以找到隐藏目录"~/.ssh"。然后复制该目录下文件id_rsa.pub的文件信息。

可以通过命令cat查看，或者直接用文本编辑器打开：
>cat ~/.ssh/id_rsa.pub

#### d. 然后将公钥信息复制到github上
<center><img src="https://selous123.github.io/assets/img/blog-gpage/gsettings.png" width="200" height="260"/></center>

<center><img src="https://selous123.github.io/assets/img/blog-gpage/gsettings2.png" width="500" height="160"/></center>

<center><img src="https://selous123.github.io/assets/img/blog-gpage/gsettings3.png" width="500" height="260"/></center>

设置成功之后，就可以实现本地git通过ssh和github连接了
<center><img src="https://selous123.github.io/assets/img/blog-gpage/gsettings4.png" width="600" height="260"/></center>

### 2. 如何搭建自己的github-page.
#### 2.1. 在github网站上新建项目，项目名称为 "用户名.github.io"

<center><img src="https://selous123.github.io/assets/img/blog-gpage/create.png" width="400" height="320"/></center>

#### 2.2. 将该项目pull到本地
> git clone git@github.com:用户名/用户名.github.io.git 

#### 2.3. 选择一个自己喜欢的模板格式，如[我的博客](selous123.github.io),找到该博客的[github repo](https://github.com/selous123/selous123.github.io),然后pull到本地
>git clone https://github.com/selous123/selous123.github.io.git

#### 2.4. 然后将文件下的除了.git外的所有文件复制到自己的目录下

<center><img src="https://selous123.github.io/assets/img/blog-gpage/file.png" width="400" height="300"/></center>

#### 2.5. 将文件push到自己的repo中
> git add -A

> git commit -m "init repo"

> git push origin master


#### 2.6. 最后就可以通过网址: 用户名.github.io访问到博客内容。
<center><img src="https://selous123.github.io/assets/img/blog-gpage/blog.png" width="700" height="350"/></center>

**也可以通过github repo的setting选项查看自己blog的状态。**

<center><img src="https://selous123.github.io/assets/img/blog-gpage/settings.png" width="700" height="140"/></center>
然后可以看到github-pages的状态如下:
<center><img src="https://selous123.github.io/assets/img/blog-gpage/gp_zt.png" width="700" height="150"/></center>

### 3. 写在后面
#### 3.1. 如何将博客私有化
##### a. 删除CNAME 文件中的内容，不然会报warning.
##### b. 配置about页面
打开 about.md文件，修改关于自己页面
<center><img src="https://selous123.github.io/assets/img/blog-gpage/psettings1.png" width="700" height="330"/></center>

##### c. 私有化首页内容
打开_config.yml文件，修改自己的个性化设置
<center><img src="https://selous123.github.io/assets/img/blog-gpage/psettings2.png" width="700" height="330"/></center>

##### d. 写博客
将_post文件夹中博客删除，创建自己新的博客。

命名格式为: YYYY-MM-DD-nameofblog.md

<center><img src="https://selous123.github.io/assets/img/blog-gpage/psettings3.png" width="700" height="250"/></center>

写完然后上传到github就可以看到博客内容了.

**注意: 网页更新比较慢，不会即时生效。有的时候也需要清理一下浏览器缓存才能正常访问到更新的内容。**

#### 3.2. 推荐编辑器--vscode



Enjoy Yourself in Blog!