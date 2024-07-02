// src/WebcamCapture.tsx
import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as faceDetection from "@tensorflow-models/face-detection";
import "@tensorflow/tfjs-backend-webgl";

const BLUR_THRESHOLD_LAPLACIAN = 100; // Adjusted threshold for Laplacian variance method
const BLUR_THRESHOLD_TENENGRAD = 2000; // Threshold for Tenengrad method
const MIN_BOX_SIZE = 200; // Minimum size for the bounding box to consider the face clear
const IMG_SIZE = 640;
const videoConstraints = {
  width: IMG_SIZE,
  height: IMG_SIZE,
  facingMode: "user",
};

declare global {
  interface Window {
    cv: any;
  }
}

const WebcamCapture: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [detector, setDetector] = useState<faceDetection.FaceDetector | null>(
    null
  );
  const [cvLoaded, setCvLoaded] = useState<boolean>(false);
  const [isImageClear, setIsImageClear] = useState<boolean>(false);

  // Load model only once
  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      const model = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        {
          runtime: "tfjs",
          modelType: "short",
          maxFaces: 1,
        }
      );
      setDetector(model);
    };

    loadModel();
  }, []);

  // Ensure OpenCV.js is loaded
  useEffect(() => {
    const checkOpenCV = setInterval(() => {
      if (window.cv) {
        setCvLoaded(true);
        clearInterval(checkOpenCV);
      }
    }, 100);

    return () => clearInterval(checkOpenCV);
  }, []);

  // Function to detect blurriness using Laplacian variance method
  const isBlurryLaplacian = useCallback((image: HTMLImageElement): boolean => {
    if (!window.cv) return true; // If OpenCV is not loaded, assume image is blurry

    const mat = window.cv.imread(image);
    const grayMat = new window.cv.Mat();
    window.cv.cvtColor(mat, grayMat, window.cv.COLOR_RGBA2GRAY);
    const laplacianMat = new window.cv.Mat();
    window.cv.Laplacian(grayMat, laplacianMat, window.cv.CV_64F);
    const mean = new window.cv.Mat();
    const stddev = new window.cv.Mat();
    window.cv.meanStdDev(laplacianMat, mean, stddev);
    const variance = Math.pow(stddev.data64F[0], 2);
    mat.delete();
    grayMat.delete();
    laplacianMat.delete();
    mean.delete();
    stddev.delete();

    console.log(`Laplacian Variance: ${variance}`); // Log variance for tuning
    return variance < BLUR_THRESHOLD_LAPLACIAN;
  }, []);

  // Function to detect blurriness using Tenengrad method
  const isBlurryTenengrad = useCallback((image: HTMLImageElement): boolean => {
    if (!window.cv) return true; // If OpenCV is not loaded, assume image is blurry

    const mat = window.cv.imread(image);
    const grayMat = new window.cv.Mat();
    window.cv.cvtColor(mat, grayMat, window.cv.COLOR_RGBA2GRAY);
    const sobelX = new window.cv.Mat();
    const sobelY = new window.cv.Mat();
    window.cv.Sobel(grayMat, sobelX, window.cv.CV_64F, 1, 0);
    window.cv.Sobel(grayMat, sobelY, window.cv.CV_64F, 0, 1);
    const sobelMat = new window.cv.Mat();
    window.cv.magnitude(sobelX, sobelY, sobelMat);
    const mean = new window.cv.Mat();
    const stddev = new window.cv.Mat();
    window.cv.meanStdDev(sobelMat, mean, stddev);
    const variance = Math.pow(stddev.data64F[0], 2);
    mat.delete();
    grayMat.delete();
    sobelX.delete();
    sobelY.delete();
    sobelMat.delete();
    mean.delete();
    stddev.delete();

    console.log(`Tenengrad Variance: ${variance}`); // Log variance for tuning
    return variance < BLUR_THRESHOLD_TENENGRAD;
  }, []);

  // Combined function to detect blurriness using both methods
  const isBlurry = useCallback(
    (image: HTMLImageElement): boolean => {
      if (isBlurryLaplacian(image)) console.log("fail Laplacian");
      else if (isBlurryTenengrad(image)) console.log("fail Tenengrad");

      return isBlurryLaplacian(image) || isBlurryTenengrad(image);
    },
    [isBlurryLaplacian, isBlurryTenengrad]
  );

  // Function to detect faces and blurriness continuously
  const detectFace = useCallback(async () => {
    if (webcamRef.current && detector && cvLoaded) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const img = new Image();
        img.src = imageSrc;
        await img.decode();

        // Check for blur using both methods
        if (isBlurry(img)) {
          console.log("Image is blurry");
          setIsImageClear(false);
          requestAnimationFrame(detectFace); // Continue detection
          return;
        }

        const detections = await detector.estimateFaces(img);

        // Ensure bounding box size
        const isClear = detections.some((detection) => {
          const box = detection.box;
          return box.width >= MIN_BOX_SIZE && box.height >= MIN_BOX_SIZE;
        });

        setIsImageClear(isClear);
        requestAnimationFrame(detectFace); // Continue detection
      }
    }
  }, [webcamRef, detector, cvLoaded, isBlurry]);

  // Start continuous face detection
  useEffect(() => {
    requestAnimationFrame(detectFace);
  }, [detectFace]);

  // Capture frame on button click
  const captureFrame = useCallback(() => {
    if (webcamRef.current && isImageClear) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImage(imageSrc);
    } else {
      alert("No clear face detected.");
    }
  }, [webcamRef, isImageClear]);

  return (
    <div>
      <h1>Manual Capture</h1>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={IMG_SIZE}
        height={IMG_SIZE}
        videoConstraints={videoConstraints}
        className={`webcam-frame ${isImageClear ? "clear" : "unclear"}`}
      />
      <button onClick={captureFrame}>Capture</button>
      {capturedImage && (
        <div>
          <h2>Face Detected!</h2>
          <img src={capturedImage} alt="Captured Frame with Face" />
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;
