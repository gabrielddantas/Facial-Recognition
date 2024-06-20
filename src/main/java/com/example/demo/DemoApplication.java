package com.example.demo;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.List;

@SpringBootApplication
public class DemoApplication {

  static {
    OpenCV.loadShared();
  }

  public static void main(String[] args) {
    SpringApplication.run(DemoApplication.class, args);

    String imgFile1 = "/home/gabrieldantas/tests/demo/src/main/resources/static/foto1.jpeg";
    String imgFile2 = "/home/gabrieldantas/tests/demo/src/main/resources/static/foto2.jpeg";
    String faceCascadeFile =
        "/home/gabrieldantas/tests/demo/src/main/resources/static/haarcascade_frontalface_alt.xml";

    CascadeClassifier faceDetector = new CascadeClassifier(faceCascadeFile);

    Mat image1 = Imgcodecs.imread(imgFile1);
    Mat image2 = Imgcodecs.imread(imgFile2);

    MatOfRect faceDetections1 = new MatOfRect();
    MatOfRect faceDetections2 = new MatOfRect();

    faceDetector.detectMultiScale(image1, faceDetections1);
    faceDetector.detectMultiScale(image2, faceDetections2);

    System.out.println(
        String.format("Detected %s faces in first image", faceDetections1.toArray().length));
    System.out.println(
        String.format("Detected %s faces in second image", faceDetections2.toArray().length));

    Imgcodecs.imwrite("output1.jpg", getRect(image1, faceDetections1));
    Imgcodecs.imwrite("output2.jpg", getRect(image2, faceDetections2));

    if (faceDetections1.toArray().length > 0 && faceDetections2.toArray().length > 0) {
      Rect face1 = faceDetections1.toArray()[0];
      Rect face2 = faceDetections2.toArray()[0];

      Mat faceROI1 = new Mat(image1, face1);
      Mat faceROI2 = new Mat(image2, face2);

      Imgproc.resize(faceROI1, faceROI1, new Size(200, 200));
      Imgproc.resize(faceROI2, faceROI2, new Size(200, 200));

      boolean isSimilar = compareFaces(faceROI1, faceROI2);
      System.out.println("Are the faces similar? " + isSimilar);
    }
  }

  private static Mat getRect(Mat image, MatOfRect faceDetections) {
    Rect img1Crop = null;
    for (Rect rect : faceDetections.toArray()) {
      Imgproc.rectangle(
          image,
          new Point(rect.x, rect.y),
          new Point(rect.x + rect.width, rect.y + rect.height),
          new Scalar(0, 255, 0));
      img1Crop = new Rect(rect.x, rect.y, rect.width, rect.height);
    }
    return new Mat(image, img1Crop);
  }

  public static boolean compareFaces(Mat face1, Mat face2) {
    Mat hist1 = new Mat();
    Mat hist2 = new Mat();

    Imgproc.cvtColor(face1, face1, Imgproc.COLOR_BGR2GRAY);
    Imgproc.cvtColor(face2, face2, Imgproc.COLOR_BGR2GRAY);

    Mat histPic1 = getHistogram(face1, hist1);
    Mat histPic2 = getHistogram(face2, hist2);

    Core.normalize(histPic1, histPic1);
    Core.normalize(histPic2, histPic2);

    double comparison = Imgproc.compareHist(histPic1, histPic2, Imgproc.CV_COMP_CORREL);
    System.out.println("Comparison: " + Math.round(comparison * 100));
    return comparison > 0.6;
  }

  private static Mat getHistogram(Mat image, Mat hist) {
    Imgproc.calcHist(
        List.of(image),
        new MatOfInt(0),
        new Mat(),
        hist,
        new MatOfInt(256),
        new MatOfFloat(0, 256));
    return hist;
  }
}
