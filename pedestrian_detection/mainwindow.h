#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<string>
#include<QSystemTrayIcon>
#include<QWizard>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


public:
    std::string detectionImgFile;

private slots:
    void readImg();//读取图片
    void detectionPedestrain();//行人检测
    void detectionShowImg();//展示检测结果
    void detectionSaveImg();//保存检测结果

    void on_action_A_2_triggered();//关于本软件
//    void openUrl(QString url);

//    void on_action_bug_triggered();

    void on_action_A_triggered();

    void on_action_bug_triggered();

    void on_action_D_triggered();

private:
    Ui::MainWindow *ui;
    QWizardPage * createPage1();
    QWizardPage * createPage2();
    QWizardPage * createPage3();
    QWizardPage * createPage4();
};

#endif // MAINWINDOW_H
