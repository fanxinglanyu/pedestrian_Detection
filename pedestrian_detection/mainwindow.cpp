#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileInfo>
#include<QFileDialog>
#include<QDebug>
#include<QImage>
#include <QByteArray>
#include<QFile>
#include<QDesktopServices>
#include<QMessageBox>
#include<QLabel>
#include<QHBoxLayout>
#include<string>

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include <QCoreApplication>

using namespace std;
using namespace cv;
using namespace cv::ml;

bool isDetection = false;//判断是否行人检测成功
bool isSaveImg = false;//判断是否存取了图片

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setFixedSize( this->width (),this->height ());
    this->setWindowTitle(tr("行人检测软件v1.0版"));
    //读取图片
    connect(ui->Button_readImg,&QPushButton::clicked,
            this, &MainWindow::readImg);
    //检测图片
    connect(ui->Button_detection,&QPushButton::clicked,
            this, &MainWindow::detectionPedestrain);
    //显示图片
    connect(ui->Button_showImg,&QPushButton::clicked,
            this,&MainWindow::detectionShowImg);
    //保存结果
    connect(ui->Button_saveImg,&QPushButton::clicked,
            this,&MainWindow::detectionSaveImg);


    //添加菜单
    //ui->mainToolBar->addAction(action_M);

//    trayicon = new QSystemTrayIcon(this);
//       connect(trayicon, SIGNAL(activated(QSystemTrayIcon::ActivationReason)), this, SLOT(onSystemTrayIconClicked(QSystemTrayIcon::ActivationReason)));
//       QIcon icon("MyICO.ico");
//       trayicon->setIcon(icon);
//       trayicon->show();

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::readImg()
{
    isDetection = false;//重置行人检测结果

    QString file_full, file_name, file_path;
           QString file_suffix,file_fullname;
           QFileInfo fi;//文件路径
           file_full = QFileDialog::getOpenFileName(this);
       //      打开文件        获取文件路径
         //  file_full = QFileDialog::getSaveFileName(this,tr("Save File"),tr("*.txt"));
       //      保存文件        获取文件路径  “Save File” 对话框名        “*.txt” 默认文件名
           fi= QFileInfo(file_full);


       //      显示文件路径
           if(fi.exists() == true){
               //file_name = fi.fileName();//文件名称
               //file_path = fi.absolutePath();//文件路径，不包含名称
                //ui->textBrowser->setText(file_path+"/"+file_name);
               file_fullname = fi.filePath();//文件全部名称
               file_suffix = fi.suffix();
              // ui->textBrowser->setText(file_fullname+"/"+file_suffix);
               if(file_suffix == "jpg" || file_suffix == "png"){
                      ui->textBrowser->setText(tr("图片文件读取成功！"));
                     detectionImgFile = file_fullname.toStdString();
                     // imageFilePath = file_fullname;
               }else{
                   ui->textBrowser->setText(tr("请选择png或者jpg格式的图片文件!"));
               }

           }else {
               ui->textBrowser->setText(tr("图片文件读取为空！"));
   }
   //        file_name = fi.fileName();
   //        file_path = fi.absolutePath();
           //ui->textBrowser->setText(file_path+"\\"+file_name);

}

///Users/macbookpro/Desktop/040419_mt_eht-array_feat_free.jpg
void MainWindow::detectionPedestrain()
{


//    Mat img = imread(detectionImgFile );
//    if(img.empty()){
//        ui->textBrowser->setText(tr("请选择图片!"));
//    }else{
//        namedWindow("行人");
//        imshow("行人图片",img);
//        waitKey(5000);
//        destroyWindow("行人");
//    }
//    QFile file(":/detectionFile/source/SVM_HOG.xml");


//正式开始
        Mat src = imread(detectionImgFile);
        if(src.empty()){
             ui->textBrowser->setText(tr("请选择需要检测的图片文件!"));
        }else{
            //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
               HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
           //    int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
               Ptr<SVM> svm = SVM::create();// 创建分类器

               //读取xml文件
//                      QString temp = QDir::currentPath();
//                      string tempname = temp.toStdString();
//                      tempname += "/SVM_HOG.xml";

                        QString temp = QApplication::applicationDirPath();
                        string tempname = temp.toStdString() +"/SVM_HOG.xml";
//                        qDebug() << QString::fromStdString(tempname);

                      svm = SVM::load(tempname);


               int svdim = svm ->getVarCount();//特征向量的维数，即HOG描述子的维数
               //支持向量的个数
               Mat svecsmat = svm ->getSupportVectors();//svecsmat元素的数据类型为float
               int numofsv = svecsmat.rows;

               Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);
               Mat svindex = Mat::zeros(1, numofsv,CV_64F);
             //  cout << "after initialize the value of alphamat is  " << alphamat.size()  << endl;

               Mat Result;
               double rho = svm ->getDecisionFunction(0, alphamat, svindex);

           //    cout << "the value of rho is  " << rho << endl;
               alphamat.convertTo(alphamat, CV_32F);//将alphamat元素的数据类型重新转成CV_32F
           //    cout << "the value of alphamat is  " << alphamat << endl;
           //    cout << "the size of alphamat is  " << alphamat.size() << endl;
           //    cout << "the size of svecsmat is  " << svecsmat.size() << endl;

               //计算-(alphaMat * supportVectorMat),结果放到resultMat中
               Result = -1 * alphamat * svecsmat;//float

           //    cout << "the value of svdim is  " << svdim << endl;

               //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
               vector<float> vec;
               //将resultMat中的数据复制到数组vec中
               for (int i = 0; i < svdim; ++i)
               {
                   vec.push_back(Result.at<float>(0, i));
               }

               vec.push_back((float)rho);

               /*********************************开始检测**************************************************/
               HOGDescriptor hog_test;
               hog_test.setSVMDetector(vec);


               vector<Rect> found, found_filtered;
               hog_test.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);

               if(found.size() > 0){
                   //cout<<"Congratulations！This picture has pedestrian."<<endl;
                   //cout<<"Congratulations！This picture has "<< found.size()<<" pedestrian."<<endl;
               ui->textBrowser->setText(tr("恭喜！这张图片有行人！"));
               }else{
               ui->textBrowser->setText(tr("抱歉，这张图片，不存在行人！"));

                  // cout<<"Sorry. This picture has no pedestrian!"<<endl;
               }

               //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
               for(int i=0; i < found.size(); i++)
               {
                   Rect r = found[i];
                   int j=0;
                   for(; j < found.size(); j++)
                       if(j != i && (r & found[j]) == r)
                           break;
                   if( j == found.size())
                       found_filtered.push_back(r);
               }


               //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
               for(int i=0; i<found_filtered.size(); i++)
               {
                   Rect r = found_filtered[i];
                   r.x += cvRound(r.width*0.1);
                   r.width = cvRound(r.width*0.8);
                   r.y += cvRound(r.height*0.07);
                   r.height = cvRound(r.height*0.8);
                   rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
               }
               QString temp1 = QApplication::applicationDirPath();
               string tempname1 = temp.toStdString() +"/ImgProcessed.jpg";

               imwrite(tempname1,src);//保存图片
               isDetection = true;//行人检测成功
               isSaveImg = true;//保存了检测的图片
        }


}

void MainWindow::detectionShowImg()
{
    if(isDetection == true){
        //获取检测后保存的图片
//        QString temp = QDir::currentPath();
//        string tempname = temp.toStdString();

        QString temp = QApplication::applicationDirPath();
        string tempname = temp.toStdString();

        tempname += "/ImgProcessed.jpg";
        Mat img = imread(tempname);
        namedWindow("检测结果",-1);
        imshow("检测结果",img);
        waitKey(0);
        destroyWindow("检测结果");
    }else{
        ui->textBrowser->setText(tr("请先进行行人检测！"));
    }

}

void MainWindow::detectionSaveImg()
{

    if(isDetection == true){
        //    //获取原始要保存图片的路径
//            QString temp = QDir::currentPath();
//            string tempname = temp.toStdString();

            QString temp = QApplication::applicationDirPath();
            string tempname = temp.toStdString();
            tempname += "/ImgProcessed.jpg";

            Mat img = imread(tempname);

        //    //获取要保存的路径

                QString file_path = QFileDialog::getExistingDirectory(this, "请选择文件路径...", "./");
                if(file_path.isEmpty())
                {
                    ui->textBrowser->setText(tr("请选择要保存的文件夹!"));
                }else{
                    ui->textBrowser->setText(file_path);

                     string detectionSaveImgFile = file_path.toStdString();
                     imwrite(detectionSaveImgFile +"/imgResult.jpg",img);
                }

    }else{
        ui->textBrowser->setText(tr("请先进行行人检测！"));
    }


//    QFileDialog* fileDialog = new QFileDialog(this);
//    fileDialog->setWindowTitle("Choose Source Directory");
//    //fd->setDirectory(buf);
//    fileDialog->setFileMode( QFileDialog::DirectoryOnly );
//    QStringList fileName;
//    if ( fileDialog->exec() == QDialog::Accepted )
//    {
//        fileName = fileDialog->selectedFiles();
//        //srcDir.setPath(fileName.at(0));
//        ui->textBrowser->append(fileName.join(","));
//    }
//    else
//    {
//        ui->textBrowser->setText(tr("请选择要保存的文件夹!"));
//    }


}




//帮助按钮功能
void MainWindow::on_action_A_2_triggered()
{
    QMessageBox::about(this,tr("关于本软件"),tr("这是一个基于HOG+SVM并用Qt编写的行人检测软件v1.0版本，欢迎使用！作者：韩江雪"));
}


QWizardPage * MainWindow::createPage1(){
    QWizardPage * page = new QWizardPage;
    page->setTitle(tr("帮助对话框"));
    page->setTitle(tr("首先，非常欢迎使用本软件！让我们赶快开始吧！\n点击“选择图片”按钮，来读入将要检测的图片。"));
    page->setButtonText(QWizard::NextButton,"下一步");
    page->setButtonText(QWizard::BackButton,"上一步");
    page->setButtonText(QWizard::CancelButton,"取消");
    page->setButtonText(QWizard::FinishButton,"完成");


    QLabel *picLabel = new QLabel;
    picLabel->setPixmap(QPixmap(":/image/images/guide1.jpg"));
    QHBoxLayout *firstLayout = new QHBoxLayout;
    firstLayout->addWidget(picLabel);
    page->setLayout(firstLayout);
    return page;
}

QWizardPage * MainWindow::createPage2(){
    QWizardPage * page = new QWizardPage;
    page->setTitle(tr("第二步"));
    page->setTitle(tr("然后点击“开始检测”按钮，来开始检测工作。检测完成后，会以文字结果呈现在屏幕的右端的文字框中。"));
    page->setButtonText(QWizard::NextButton,"下一步");
    page->setButtonText(QWizard::BackButton,"上一步");
    page->setButtonText(QWizard::CancelButton,"取消");
    page->setButtonText(QWizard::FinishButton,"完成");

    QLabel *picLabel = new QLabel;//设置图片
    picLabel->setPixmap(QPixmap(":/image/images/guide2.jpg"));
    QHBoxLayout *secondLayout = new QHBoxLayout;
    secondLayout->addWidget(picLabel);
    page->setLayout(secondLayout);

    return page;
}

QWizardPage * MainWindow::createPage3(){
    QWizardPage * page = new QWizardPage;
    page->setTitle(tr("第三步"));
    page->setTitle(tr("如果需要图片的检测结果，请点击“结果展示”按钮按钮。"));
    page->setButtonText(QWizard::NextButton,"下一步");
    page->setButtonText(QWizard::BackButton,"上一步");
    page->setButtonText(QWizard::CancelButton,"取消");
    page->setButtonText(QWizard::FinishButton,"完成");

    QLabel *picLabel = new QLabel;//设置图片
    picLabel->setPixmap(QPixmap(":/image/images/guide3.jpg"));
    QHBoxLayout *thirdLayout = new QHBoxLayout;
    thirdLayout->addWidget(picLabel);
    page->setLayout(thirdLayout);
    return page;
}

QWizardPage * MainWindow::createPage4(){
    QWizardPage * page = new QWizardPage;
    page->setTitle(tr("第四步"));
    page->setTitle(tr("如果你需要保存检测后的结果图片，请点击“保存结果”按钮。"));
    page->setButtonText(QWizard::NextButton,"下一步");
    page->setButtonText(QWizard::BackButton,"上一步");
    page->setButtonText(QWizard::CancelButton,"取消");
    page->setButtonText(QWizard::FinishButton,"完成");

    QLabel *picLabel = new QLabel;//设置图片
    picLabel->setPixmap(QPixmap(":/image/images/guide4.jpg"));
    QHBoxLayout *fourthLayout = new QHBoxLayout;
    fourthLayout->addWidget(picLabel);
    page->setLayout(fourthLayout);
    return page;
}

void MainWindow::on_action_A_triggered()
{
    QWizard wizard(this);
    wizard.setOption(QWizard::NoBackButtonOnStartPage );//设置第一页没有上一步按钮
    wizard.setWizardStyle( QWizard::ModernStyle );//设置上一步下一步等按钮的显示格式
  //  wizard.setWindowTitle(tr("帮助对话框"));//设置框的标题
    wizard.addPage(createPage1());
    wizard.addPage(createPage2());
    wizard.addPage(createPage3());
    wizard.addPage(createPage4());
    wizard.exec();
}

void MainWindow::on_action_bug_triggered()
{
//    QLabel *linkLabel = new QLabel();
//    linkLabel->setOpenExternalLinks(true);
//    linkLabel->setText("<a href=\"https://blog.csdn.net/qq_33375598\">linkLabelTest");
//    linkLabel->setText("<a style='color: green;' href=\"http://www.cnblog.com/fron_csl\">linkLabel");

    QMessageBox::about(this,tr("帮助我们做的更好！"),tr("谢谢你，帮助我们做的更好！\nbug报告网站：https://blog.csdn.net/qq_33375598"));

}

void MainWindow::on_action_D_triggered()
{
    QString tempname = QDir::currentPath();
    tempname += "/ImgProcessed.jpg";

    QFile fi(tempname);
    if(isSaveImg == true){
        fi.remove();
        QMessageBox::information(this,tr("删除对话框"),tr("已经成功删除检索后的图片！"),QMessageBox::Ok);
    }else{
        QMessageBox::critical(this,tr("错误对话框"),tr("没有存储任何检索后的图片！"),QMessageBox::Ok);
    }

}
