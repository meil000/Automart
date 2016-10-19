package test;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.imageio.stream.ImageOutputStream;
import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class opencvTest{
	
	public static void main(String[] args) {
	    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	    
	    TemplateMaching macher = new TemplateMaching();
	    //设置原图
	    macher.setSource("H:/temp/3.png");
	    //设置要匹配的图
	    macher.setDst("H:/temp/11.png");
	    //设置输出结果的图
	    macher.setOutput("H:/temp/333.png");
	    
	    macher.process();
	}
	
//	public static void main(String[] args) {
//		String srcImgFile = "H:/temp/2.png";
//		String expImgFile = "H:/temp/3.png";
//		File imageFile = new File(srcImgFile);
//		BufferedImage oImage = null;
//		try {
//			oImage = ImageIO.read(imageFile);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		int srcWidth = oImage.getWidth();
//		int srcHeight = oImage.getHeight();
//		zoomImage(srcImgFile, srcWidth*2, srcHeight*2, expImgFile, "png");
//	}
	 
	//缩小、放大图片
	public static boolean zoomImage(String imageFile, int width, int height, String saveFile, String imageFormat){
		try{
			File expectImageFile = new File(saveFile);
			if (expectImageFile.exists())
			{
				FileUtils.forceDelete(new File(saveFile));
			}
			
			BufferedImage originalImage = ImageIO.read(new File(imageFile));
			BufferedImage newImage = new BufferedImage(width,height,originalImage.getType());
			Graphics g = newImage.getGraphics();
		    g.drawImage(originalImage, 0,0,width,height,null);
		    g.dispose();
		       
		    ByteArrayOutputStream byteOut= new ByteArrayOutputStream();
		    ImageOutputStream iamgeOut = ImageIO.createImageOutputStream(byteOut);
		    ImageIO.write(newImage, imageFormat, iamgeOut); 
		    FileUtils.writeByteArrayToFile(new File(saveFile), byteOut.toByteArray());
		    byteOut.close();
		    return true;
		}
		catch(Exception e){
			System.out.println("Failed to zoom iamge" + "\n" + e);
		}
			return false;
	}
}

class TemplateMaching {
	
	private String sourcePath,dstPath,outputPath;
	private Mat source,dst;
	//原图片
	public void setSource(String picPath){
	    this.sourcePath = picPath;
	}
	//需要匹配的部分
	public void setDst(String picPath){
	    this.dstPath = picPath;
	}
	//输入结果的图片
	public void setOutput(String picPath){
	    this.outputPath = picPath;
	}
	
	//处理，生成结果图
	public void process(){
	    //将文件读入为OpenCV的Mat格式
	    source = Imgcodecs.imread(sourcePath);
	    dst = Imgcodecs.imread(dstPath);
	    //创建于原图相同的大小，储存匹配度
	    Mat result = Mat.zeros(source.rows(),source.cols(),CvType.CV_32FC1);
	    //调用模板匹配方法
	    Imgproc.matchTemplate(source, dst, result,Imgproc.TM_SQDIFF);
	    //规格化
	    Core.normalize(result, result, 0, 1,Core.NORM_MINMAX, -1);
	    //获得最可能点，MinMaxLocResult是其数据格式，包括了最大、最小点的位置x、y
	    MinMaxLocResult mlr = Core.minMaxLoc(result);
	    Point matchLoc = mlr.minLoc;
	    double matchMaxValue = mlr.maxVal;
	    double matchMinValue = mlr.minVal;
	    System.out.println("matchMaxValue is: " + matchMaxValue + "; matchMinValue is: " + matchMinValue +".");
	    //在原图上的对应模板可能位置画一个绿色矩形
	    Imgproc.rectangle(source, matchLoc, new Point(matchLoc.x + dst.width(),matchLoc.y + dst.height()),new Scalar(0,255,0));
	    //将结果输出到对应位置
	    File expectImageFile = new File(outputPath);
		if (expectImageFile.exists())
		{
			try {
				FileUtils.forceDelete(new File(outputPath));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	    Imgcodecs.imwrite(outputPath,source);
	}
}
