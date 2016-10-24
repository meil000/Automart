package test;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
	    String srcFilePath = "H:/temp/src/";
	    String dstFilePath = "H:/temp/dst/";
	    String outFilePath = "H:/temp/out/";
	    
	    //如果文件夹不存在则创建
	    File file = new File(srcFilePath);
	    if(!file.exists() && !file.isDirectory()){       
	        file .mkdir();    
	    }
	    file = new File(dstFilePath);
	    if(!file.exists() && !file.isDirectory()){       
	        file .mkdir();    
	    }
	    file = new File(outFilePath);
	    if(!file.exists() && !file.isDirectory()){       
	        file .mkdir();    
	    }
	    
	    String fileType = "png";
	    String srcFileName = "1";
	    String outFileName = "111";
	    String expSimilarity = "50%";
	    //设置原图
	    macher.setSource(srcFilePath + srcFileName + "." + fileType);
	    //设置要匹配的图的路径
	    macher.setDstPath(dstFilePath);
	    //设置输出结果的图
	    macher.setOutput(outFilePath + outFileName + "." + fileType);
	    
	    macher.matchDstImg(Double.valueOf(expSimilarity.substring(0, (expSimilarity.length() - 1)))/100);
	}
	
//	public static void main(String[] args) {
//		String srcImgFile = "E:/temp/opencv/tmp_new_2.jpg";
//		String expImgFile = "E:/temp/opencv/tmp_new_4.jpg";
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
//		zoomImage(srcImgFile, srcWidth/2, srcHeight/2, expImgFile, "jpg");
//	}
//	 
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
	
	private String sourcePath,tmpSourcePath,dstPath,outputPath,dstImgName;
	private Mat source,dst;
	private List<Mat> matchImgList = new ArrayList<Mat>();
	private List<String> matchImgNameList = new ArrayList<String>();
	private List<Mat> ignoreMatchImgList = new ArrayList<Mat>();
	//原图片
	public void setSource(String picPath){
	    this.sourcePath = picPath;
	}
	//需要匹配的图片路径
	public void setDstPath(String picPath){
	    this.dstPath = picPath;
	}
	//输出结果的图片
	public void setOutput(String picPath){
	    this.outputPath = picPath;
	}
	
	//根据相似度在图中查找是否存在预期图像
	public void matchDstImg(double similarity){
	    //将文件读入为OpenCV的Mat格式
	    source = Imgcodecs.imread(sourcePath);
	    //创建于原图相同的大小，储存匹配度
	    Mat result;
	    readImageFiles();
	    for(Mat dstImg : matchImgList){
	    	dst = dstImg;
	    	result = Mat.zeros(source.rows() - dst.width(), source.cols() - dst.height(), CvType.CV_32FC1);
	    	//调用模板匹配方法
		    Imgproc.matchTemplate(source, dst, result,Imgproc.TM_CCOEFF_NORMED);
		    //规格化
//		    Core.normalize(result, result, 0, 1,Core.NORM_MINMAX, -1);
		    //根据匹配结果找到符合要求的图片，并画框
		    int num = 1;    //符合相似度的匹配次数，初始为1
		    findSimilar(result, similarity, num, true);
	    }
	    
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
	
	//查找符合匹配度的图片
	public void findSimilar(Mat result, double similarity, int num, boolean isMult){
		//获得最可能点，包括改点的坐标，相似度，MinMaxLocResult是其数据格式，包括了最大、最小点的位置x、y
	    MinMaxLocResult mlr = Core.minMaxLoc(result);
	    Point matchLoc = mlr.maxLoc;
	    double maxLocX = mlr.maxLoc.x;
	    double maxLocY = mlr.maxLoc.y;
	    double minVal = mlr.minVal;
	    double maxVal = mlr.maxVal;
	    //在原图上的对应模板可能位置画一个绿色矩形
	    if(maxVal >= similarity){
	    	//如果匹配成功，获取当前匹配图片的名称和匹配成功的次数
	    	System.out.println("匹配到第" + num + "个与" + matchImgNameList.get(0) + "相似的图像坐标是：[" + maxLocX + "," + maxLocY + "]; minVal = " + minVal + "; maxVal = " + maxVal + "。");
	    	Imgproc.rectangle(source, matchLoc, new Point(matchLoc.x + dst.width(),matchLoc.y + dst.height()),new Scalar(0,255,0));
	    }else{
	    	//如果匹配结束或匹配不到预期结果，跳过当前匹配图片的名称
	    	matchImgNameList.remove(0);
	    }
		//将已经成功匹配找出的图像部分，从结果中剔除，避免下一次匹配时再次被找到
		int startX = (int)matchLoc.x - dst.width() + 1;
		int startY = (int)matchLoc.y - dst.height() + 1;
		int endX = (int)matchLoc.x + dst.width() - 1;
		int endY = (int)matchLoc.y + dst.height() - 1;
		if(startX < 0){
			startX =0;
		}
		if(startY < 0){
			startY = 0;
		}
		if(endX > result.width() - 1){
			endX = result.width() - 1;
		}
		if(endY > result.height() - 1){
			endY = result.height();
		}
		int y, x;
		for(y = startY; y < endY; y++){
			for(x = startX; x < endX; x++){
				result.put(y, x, minVal);
			}
		}
		num++;
		//如果匹配多个，递归查找下一个符合匹配要求的图像起点坐标,并给图像画框
		if(maxVal >= similarity && isMult == true){
			findSimilar(result, similarity, num, isMult);
		}
		else{
			return;
		}
	}
	
	//读取并保存预期匹配的图像
	public void readImageFiles(){
		File file = new File(dstPath);
		File [] dstImgFiles = file.listFiles();
		for(File dstImg : dstImgFiles){
			dstImgName = dstImg.getName();
			dst = Imgcodecs.imread(dstPath + dstImgName);
			matchImgNameList.add(dstImgName);
			matchImgList.add(dst);
		}
	}
}
