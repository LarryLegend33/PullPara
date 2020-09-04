using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using NationalInstruments.Vision.Acquisition.Imaq;
using NationalInstruments.Vision;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Runtime.InteropServices;
using System.Threading;
using System.IO.Ports;
using System.IO;
using System.Threading.Tasks.Dataflow;

namespace PullPara
{
    class Program
    {
        public class CamData
        {
            public Point fishcoord;
            public Mat roi;
            public uint buffernumber, jay;
            public byte[] pix_for_stim;
            public CamData(Mat roi_input, Point fishXY, uint buffer, uint j_, byte[] stmpix)
            {
                jay = j_;
                fishcoord = fishXY;
                roi = roi_input;
                buffernumber = buffer;
                pix_for_stim = stmpix;
            }
        }

        static void Main(string[] args)
        {
            SerialPort pyboard = new SerialPort("COM6", 115200);
            pyboard.Open();
            pyboard.WriteLine("import paramove\r");
            var options = new DataflowBlockOptions();
            options.BoundedCapacity = 10;
            var pipe_buffer = new BufferBlock<CamData>(options);
            bool foundfish = false;
            int l_or_r = 0; 
            MCvScalar gray = new MCvScalar(128, 128, 128);
            int roidim = 80;
            string camera_id = "img0"; //this is the ID of the NI-IMAQ board in NI MAX. 
            var _session = new ImaqSession(camera_id);

            String camerawindow = "Camera Window";
            CvInvoke.NamedWindow(camerawindow);
            int frameWidth = 1280;
            int frameHeight = 1024;
            uint bufferCount = 3;
            uint buff_out = 0;
            int numchannels = 1;
            ContourProperties fishcontour = new ContourProperties();
            System.Drawing.Size framesize = new System.Drawing.Size(frameWidth, frameHeight);
            System.Drawing.Size roi_size = new System.Drawing.Size(roidim, roidim);
            Mat cvimage = new Mat(framesize, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
            Mat modeimage = new Mat(framesize, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
            Mat modeimage_roi = new Mat(roi_size, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
            byte[,] data_2D = new byte[frameHeight, frameWidth];
            byte[,] data_2D_roi = new byte[roidim, roidim];
            byte[,] imagemode = new byte[frameHeight, frameWidth];
            ImaqBuffer image = null;
            List<byte[,]> imglist = new List<byte[,]>();
            ImaqBufferCollection buffcollection = _session.CreateBufferCollection((int)bufferCount, ImaqBufferCollectionType.VisionImage);
            _session.RingSetup(buffcollection, 0, false);
            _session.Acquisition.AcquireAsync();

            imglist = GetImageList(_session, 5000, 400);
            imagemode = FindMode(imglist);
            modeimage.SetTo(imagemode);
            imglist.Clear();
            CvInvoke.Imshow(camerawindow, modeimage);
            CvInvoke.WaitKey(0);
            Point f_center = new Point();
            Mat cv_roi = new Mat(roi_size, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
            image = _session.Acquisition.Extract((uint)0, out buff_out);
            uint j = buff_out;
            Console.WriteLine("j followed by buff_out");
            Console.WriteLine(j.ToString());
            Console.WriteLine(buff_out.ToString());
            while (true)
            {
                image = _session.Acquisition.Extract(j, out buff_out);
                data_2D = image.ToPixelArray().U8;
                cvimage.SetTo(data_2D);
        
                if (foundfish)
                {
                    modeimage_roi.SetTo(SliceROI(imagemode, f_center.X, f_center.Y, roidim));
                    data_2D_roi = SliceROI(data_2D, f_center.X, f_center.Y, roidim);
                    cv_roi = new Mat(roi_size, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
                    cv_roi.SetTo(data_2D_roi);
                    fishcontour = FishContour(cv_roi, modeimage_roi);
                    if (fishcontour.height != 0)
                    {
                        f_center.X = (int)fishcontour.center.X + f_center.X - roidim / 2;  // puts ROI coords into full frame coords
                        f_center.Y = (int)fishcontour.center.Y + f_center.Y - roidim / 2;
                    }

                    else
                    {
                        foundfish = false;
                    }
                }
                if (!foundfish)                
                {
                    fishcontour = FishContour(cvimage, modeimage);
                    if (fishcontour.height != 0)
                    {
                        f_center.X = (int)fishcontour.center.X;
                        f_center.Y = (int)fishcontour.center.Y;
//                        foundfish = true;
                        data_2D_roi = SliceROI(data_2D, f_center.X, f_center.Y, roidim);
                        cv_roi = new Mat(roi_size, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
                        cv_roi.SetTo(data_2D_roi);                        
                    }
                    else
                    {
                        foundfish = false;
                        cv_roi = new Mat(roi_size, Emgu.CV.CvEnum.DepthType.Cv8U, numchannels);
                        cv_roi.SetTo(gray); //in movie indicates that program lost the fish on this frame
                   
                        if (j % 25 == 0)
                        {
                            CvInvoke.Imshow(camerawindow, cvimage);
                            CvInvoke.WaitKey(1);
                            Console.WriteLine("Missed Fish");
                            Console.WriteLine(fishcontour.height);
                        }
                        j = buff_out + 1;
                        continue; 
                    }
                }

                if (fishcontour.com.Y > fishcontour.center.Y)
                {
//                   pyboard.WriteLine("paramove.pull_up()\r");
                    l_or_r = 1;

                }
                else if (fishcontour.com.Y < fishcontour.center.Y)
                {
// pyboard.WriteLine("paramove.pull_down()\r");
                    l_or_r = 0;
                }
                // PROBABLY MAKE THIS SO IT DOESNT DRAW DURING A STIMULUS
                if (j % 25 == 0)
                {
                    if (l_or_r == 0)
                    {
                        pyboard.WriteLine("paramove.pull_up()\r");
                        CvInvoke.Circle(cvimage, new Point(f_center.X, f_center.Y), 20, new MCvScalar(0, 0, 0));
//                        CvInvoke.Circle(cvimage, new Point(f_center.X - roidim / 2 + fish_head.X, f_center.Y - roidim / 2 + fish_head.Y), 4, new MCvScalar(255,0,0));
                        Console.WriteLine(fishcontour.height);
                    }
                    else if (l_or_r == 1)
                    {
                        pyboard.WriteLine("paramove.pull_down()\r");
                        CvInvoke.Circle(cvimage, new Point(f_center.X, f_center.Y), 20, new MCvScalar(255, 0, 0));
                        Console.WriteLine(fishcontour.height);
                    }
                  //  CvInvoke.Imshow(camerawindow, cvimage);
                  //  CvInvoke.WaitKey(1);
                }
                j = buff_out + 1;
            }


        }

        public struct ContourProperties
        {
            public Point center;
            public int height;
            public int width;
            public Point com;
        }

        static Point DarkestPixLocation(byte[,] fish_roi)
        {
            int tempmax = 0;
            int max_x = 0;
            int max_y = 0;
            Point xymax = new Point();
            for (int x = 0; x < fish_roi.GetLength(0) - 1; x++)
            {
                for (int y = 0; y < fish_roi.GetLength(1); y++)
                {
                    if (fish_roi[x, y] + fish_roi[x + 1, y] > tempmax)
                    {
                        tempmax = fish_roi[x, y] + fish_roi[x + 1, y];
                        max_x = x;
                        max_y = y;
                    }
                }  
            }
            xymax.X = max_x;
            xymax.Y = max_y;
            return xymax;
        }
              

        static List<byte[,]> GetImageList(ImaqSession ses, int numframes, int mod)
        {

            int frheight = 1024;
            int frwidth = 1280;
            List<byte[,]> avg_imglist = new List<byte[,]>();
            byte[,] avg_data_2D = new byte[frheight, frwidth];
            uint buff_out = 0;
            ImaqBuffer image = null;
            for (uint i = 0; i < numframes; i++)
            {
                image = ses.Acquisition.Extract((uint)0, out buff_out);
                avg_data_2D = image.ToPixelArray().U8;
                if (i % mod == 0)
                {
                    byte[,] avgimage_2D = new byte[frheight, frwidth];
                    Buffer.BlockCopy(avg_data_2D, 0, avgimage_2D, 0, avg_data_2D.Length);
                    avg_imglist.Add(avgimage_2D);
                }
            }
            return avg_imglist;
        }

        static ContourProperties FishContour(Mat image_raw, Mat background)
        {
            bool fishcont_found = false;
            Size frsize = new Size(image_raw.Width, image_raw.Height);
            Mat image = new Mat(frsize, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            ContourProperties contprops = new ContourProperties();
            ThresholdType ttype = 0;
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.AbsDiff(image_raw, background, image);
// This should be 30 as the LB. Switched to 20 to see if i could pick up paramecia. 
            CvInvoke.Threshold(image, image, 10, 255, ttype);
// UNCOMMENT IF YOU WANT TO SHOW THRESHOLDED IMAGE
            String camerawindow = "Camera Window";
            CvInvoke.NamedWindow(camerawindow);
            CvInvoke.Imshow(camerawindow, image);
            CvInvoke.WaitKey(1);
            CvInvoke.FindContours(image, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxNone);
            int fish_contour_index = 0;
            Rectangle bounding_rect = new Rectangle();
            for (int ind = 0; ind < contours.Size; ind++)
            {
                bounding_rect = CvInvoke.BoundingRectangle(contours[ind]);
                if (bounding_rect.Width > bounding_rect.Height)
                {
                    contprops.height = bounding_rect.Width;
                }
                else
                {
                    contprops.height = bounding_rect.Height;
                }
                if (contprops.height < 50 && contprops.height > 25)
                {
                    fish_contour_index = ind;
                    fishcont_found = true;
                    break;
                }
            }
            if (fishcont_found)
            {
                var contourCenter = new Point();
                var contourCOM = new Point();
                MCvMoments com = new MCvMoments();
                com = CvInvoke.Moments(contours[fish_contour_index]);
                contourCOM.X = (int)(com.M10 / com.M00);
                contourCOM.Y = (int) (com.M01 / com.M00);
                contourCenter.X = (int)(bounding_rect.X + (float)bounding_rect.Width / (float)2);
                contourCenter.Y = (int)(bounding_rect.Y + (float)bounding_rect.Height / (float)2);
                contprops.center = contourCenter;                
                contprops.com = contourCOM;
            }
            else
            {
                Console.WriteLine(contprops.com);
                Console.WriteLine(contprops.height);
                Console.WriteLine("no contours");
            }
            return contprops;
        }

        static byte[,] SliceROI(byte[,] rawdata, int centerX, int centerY, int dimension)
        {
            byte[,] roi = new byte[dimension, dimension];
            int roirow = 0;
            int roicol = 0;
            int half_roi_dim = dimension / 2;
            for (int rowind = centerY - half_roi_dim; rowind < centerY + half_roi_dim; rowind++)
            {
                for (int colind = centerX - half_roi_dim; colind < centerX + half_roi_dim; colind++)
                {
                    if (rowind >= 0 && colind >=0 && rowind < rawdata.GetLength(0) && colind < rawdata.GetLength(1))
                    {                                              
                        roi[roirow, roicol] = rawdata[rowind, colind];
                    }
                    else
                    {
                        roi[roirow, roicol] = 0;
                    }
                    roicol++;
                }
                roirow++;
                roicol = 0;
            }
            return roi;


        }

        static byte[,] FindMode(List<byte[,]> backgroundimages)
        {
            byte[,] image = backgroundimages[0];
            byte[,] output = new byte[image.GetLength(0), image.GetLength(1)];
            uint[] pixelarray = new uint[backgroundimages.Count];
            for (int rowind = 0; rowind < image.GetLength(0); rowind++)
            {
                for (int colind = 0; colind < image.GetLength(1); colind++)
                {
                    int background_number = 0;
                    foreach (byte[,] background in backgroundimages)
                    {
                        if (rowind == colind && rowind % 100 == 0)
                        {
                            Console.WriteLine(background[rowind, colind]); // This gives pixel vals of a line down the diagonal of the images for all images in modelist. //Values are unique indicating that the copy method is working.  
                        }
                        pixelarray[background_number] = background[rowind, colind];
                        // get mode of this. enter it as the value in output. 
                        background_number++;
                    }
                    uint mode = pixelarray.GroupBy(i => i)
                              .OrderByDescending(g => g.Count())
                              .Select(g => g.Key)
                              .First();
                    output[rowind, colind] = (byte)mode;

                }
            }
            Console.WriteLine("Done");
            return output;
        }
    }
}
