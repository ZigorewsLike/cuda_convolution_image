using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Threading;
using System.Drawing.Imaging;

namespace cuda_convolution_UI
{
    public partial class Form1 : Form
    {
        [DllImport("cudaFanc64.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern int getCudaDev();
        [DllImport("cudaFanc64.dll", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr calcConvolutionCuda(int N, int M, IntPtr img, IntPtr conv, int cN);

        private int image_width;
        private int image_height;
        private List<NumericUpDown> numud_list = new List<NumericUpDown>();
        private int default_cach = 144;
        private int cache_size = 144;
        public ImagePix unlock_img = new ImagePix(1, 1);
        public Form1()
        {
            InitializeComponent();
            stFooterChild1.Text = "Cuda devices: " + getCudaDev().ToString() + " |";
            if (getCudaDev() < 1)
            {
                MessageBox.Show("Not CUDA supporting", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    NumericUpDown numbUD = new NumericUpDown();
                    numbUD.Top = i * 20 + 25;
                    numbUD.Width = 45;
                    numbUD.Minimum = -numbUD.Maximum;
                    numbUD.Left = (numbUD.Width + 20) * j;
                    numud_list.Add(numbUD);
                    if (i == 1 && j == 1) numbUD.Value = 1;
                    this.panel1.Controls.Add(numbUD);
                }
            }
        }

        private void btnU_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog = new OpenFileDialog();
                dialog.Filter = "Изображения | *.jpg; *.png; *.jpeg; *.bmp | Все файлы | *.*";

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    pbOriginal.ImageLocation = dialog.FileName;
                }
            }
            catch (Exception)
            {
                MessageBox.Show("Not load image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        /// <summary>
        /// Применение свёртки к изображению
        /// </summary>
        /// <param name="conv_array">Массив Flatten матрицы свёртки</param>
        /// <param name="img">Массив Flatten изображения</param>
        public ImagePix apply_conv_matrix(int[] conv_array, ImagePix img)
        {
            Stopwatch stopWatch2 = new Stopwatch();
            stopWatch2.Start();
            ImagePix new_img = new ImagePix(image_width, image_height);
            for (int chl = 0; chl < 3; chl++)
            {
                for (int new_height = cache_size; new_height < image_height + cache_size - 1; new_height += cache_size)
                {
                    //MessageBox.Show("H:" + new_height.ToString());
                    if (new_height > image_height) new_height = image_height;
                    for (int new_width = cache_size; new_width < image_width + cache_size - 1; new_width += cache_size)
                    {
                        if (new_width > image_width) new_width = image_width;
                        //MessageBox.Show(new_width.ToString());
                        int[] img_array = new int[cache_size * cache_size];

                        for (int i = new_width - cache_size; i < new_width; i++)
                        {
                            for (int j = new_height - cache_size; j < new_height; j++)
                            {
                                if (chl == 0) img_array[1 * ((i % cache_size) * cache_size + (j % cache_size))] = img.GetPixel(i, j).R;
                                else if (chl == 1) img_array[1 * ((i % cache_size) * cache_size + (j % cache_size))] = img.GetPixel(i, j).G;
                                else img_array[1 * ((i % cache_size) * cache_size + (j % cache_size))] = img.GetPixel(i, j).B;
                            }
                        }
                        GCHandle handle_image = GCHandle.Alloc(img_array, GCHandleType.Pinned);
                        GCHandle handle_conv = GCHandle.Alloc(conv_array, GCHandleType.Pinned);
                        try
                        {
                            IntPtr pointer_img = handle_image.AddrOfPinnedObject();
                            IntPtr pointer_conv = handle_conv.AddrOfPinnedObject();
                            //MessageBox.Show("Pointer is writed");

                            //IntPtr c = calcConvolutionCuda(cache_size, cache_size, pointer_img, pointer_conv, (int)Math.Sqrt(conv_array.Length));
                            int[] res = new int[cache_size * cache_size];
                            IntPtr c;
                            try
                            {
                                //Console.WriteLine("FUNCTION RUN");
                                c = calcConvolutionCuda(cache_size, cache_size, pointer_img, pointer_conv, (int)Math.Sqrt(conv_array.Length));
                                //c = calcConvolutionCuda(cache_size, cache_size, img_array, conv_array, (int)Math.Sqrt(conv_array.Length));
                            }catch(System.AccessViolationException e)
                            {
                                Console.WriteLine(e.Message);
                                return null;
                            }
                            
                            Marshal.Copy(c, res, 0, cache_size * cache_size);
                            for (int i = (new_width - 1) / cache_size * cache_size; i < new_width; i++)
                            {
                                for (int j = (new_height - 1) / cache_size * cache_size; j < new_height; j++)
                                {
                                    int r = res[i % cache_size * cache_size + (j % cache_size)];
                                    Color cl = new_img.GetPixel(i, j);
                                    if (chl == 0) new_img.SetPixel(i, j, Color.FromArgb(r, cl.G, cl.B));
                                    else if (chl == 1) new_img.SetPixel(i, j, Color.FromArgb(cl.R, r, cl.B));
                                    else new_img.SetPixel(i, j, Color.FromArgb(cl.R, cl.G, r));
                                }
                            }
                        }
                        finally
                        {
                            if (handle_image.IsAllocated)
                            {
                                handle_image.Free();
                            }
                            if (handle_conv.IsAllocated)
                            {
                                handle_conv.Free();
                            }
                        }
                    }

                }
            }
            stopWatch2.Stop();
            TimeSpan ts2 = stopWatch2.Elapsed;
            stFooterChild4.Text = "time edit img: " + (ts2.Seconds*1000 + ts2.Milliseconds).ToString() + "ms";
            stFooterChild2.Text = "satus: img edit";
            lblLoading.Visible = false;
            btnAply.Enabled = true;
            this.Update();
            return new_img;
        }

        private void pbOriginal_LoadCompleted(object sender, AsyncCompletedEventArgs e)
        {
            unlock_img.Dispose();
            
            lblPreLoading.Visible = true;
            lblPreLoading.Text = "ЗАГРУЗКА ...";
            stFooterChild2.Text = "satus: img loading";
            this.Update();
            btnAply.Enabled = true;
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            lblImgWidth.Text = "width: " + pbOriginal.Image.Width + " height: " + pbOriginal.Image.Height;
            image_width = pbOriginal.Image.Width;
            image_height = pbOriginal.Image.Height;

            unlock_img = new ImagePix(image_width, image_height);

            if (image_height < cache_size || image_width < cache_size)
            {
                cache_size = Math.Min(image_height, image_width);
            }
            else if (cache_size != default_cach) cache_size = default_cach;

            Bitmap img = new Bitmap(pbOriginal.Image);
            pbResault.Image = img;
            lblPreLoading.Text = "ПОДГОТОВКА ...";
            this.Update();
            for (int new_height = 0; new_height < image_height; new_height += 1)
            {

                for (int new_width = 0; new_width < image_width; new_width += 1)
                {
                    unlock_img.SetPixel(new_width, new_height, img.GetPixel(new_width, new_height));
                }
            }
            Console.WriteLine(unlock_img.GetPixel(4, 4).R + "; " + unlock_img.width);
            lblPreLoading.Visible = false;
            stFooterChild2.Text = "satus: img load - OK";
            this.Update();
            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            stFooterChild3.Text = "time load img: " + (ts.Seconds * 1000 + ts.Milliseconds).ToString() + "ms";
        }

        private void Form1_ClientSizeChanged(object sender, EventArgs e)
        {
            lblPreLoading.Width = pbOriginal.Width = this.Width / 2 - 40;
            pbOriginal.Height = this.Height - 120 - panel1.Height;
            btn_Save.Left = lblLoading.Left = pbResault.Left = this.Width / 2 + 20;
            lblLoading.Width =  pbResault.Width = this.Width / 2 - 40;
            pbResault.Height = this.Height - 120 - panel1.Height;
            
            btnAply.Left = this.Width / 2 - 27;
            btnAply.Top = pbOriginal.Height / 2 + pbOriginal.Top - (btnAply.Height / 2);
            panel2.Top = panel1.Top = this.Height - panel1.Height - 70;
            lblPreLoading.Top = lblLoading.Top = pbOriginal.Height / 2 + pbOriginal.Top - (lblLoading.Height / 2);
        }

        private void pbResault_MouseMove(object sender, MouseEventArgs e)
        {
            float pos_x = (float)e.Location.X / (float)pbResault.Width * (float)image_width;
            float pos_y = (float)e.Location.Y / (float)pbResault.Height * (float)image_height;
            stFooterChild5.Text = "| mouse pos: " + ((int)pos_x).ToString() + ":" + ((int)pos_y).ToString();
        }

        private void nudConvSize_ValueChanged(object sender, EventArgs e)
        {
            int val = (int)nudConvSize.Value;
            numud_list.ForEach(delegate (NumericUpDown numbUD) {
                this.panel1.Controls.Remove(numbUD);
            });
            numud_list.Clear();

            for (int i = 0; i < val; i++)
            {
                for (int j = 0; j < val; j++)
                {
                    NumericUpDown numbUD = new NumericUpDown();
                    numbUD.Top = i * 20 + 25;
                    numbUD.Width = 45;
                    numbUD.Minimum = -numbUD.Maximum;
                    numbUD.Left = (numbUD.Width + 20) * j;
                    if (i == (val / 2) && j == (val / 2) && val % 2 == 1) numbUD.Value = 1;
                    numud_list.Add(numbUD);
                    this.panel1.Controls.Add(numbUD);
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            lblLoading.Visible = true;
            btnAply.Enabled = false;
            this.Update();
            int val = (int)nudConvSize.Value;
            int[] conv_matrix = new int[val * val];
            for (int i = 0; i < val * val; i++)
            {
                conv_matrix[i] = (int)numud_list.ElementAt(i).Value;
            }
            //Bitmap img = new Bitmap(pbOriginal.Image);
            ImagePix n_img =  apply_conv_matrix(conv_matrix, unlock_img);
            pbResault.Image = n_img.bitmap;
            //img.Dispose();
        }

        private void btn_Save_Click(object sender, EventArgs e)
        {
            try
            {
                SaveFileDialog dialog = new SaveFileDialog();
                dialog.Filter = "Изображения | *.jpg; *.png; *.jpeg; *.bmp | Все файлы | *.*";

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    pbResault.Image.Save(dialog.FileName);
                }
            }
            catch (Exception)
            {
                MessageBox.Show("Not load image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        /// <summary>
        /// Генерация примера по массиву
        /// </summary>
        /// <param name="conv_array">Матрицы Flatten свёртки</param>
        public void generate_conv(int[] conv_array)
        {
            int val = (int)Math.Sqrt(conv_array.Length);
            nudConvSize.Value = val;
            for (int i = 0; i < val * val; i++)
            {
                numud_list.ElementAt(i).Value = conv_array[i];
            }
        }

        private void lblExample1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            generate_conv(new int[9] { 1, 0, -1, 2, 0, -2, 1, 0, -1 });
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            generate_conv(new int[9] { -2, -1, 0, -1, 1, 1, 0, 1, 2 });
        }

        private void linkLabel2_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            //generate_conv(new int[25] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 });
            generate_conv(new int[25] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
        }

        private void linkLabel3_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            generate_conv(new int[25] { 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 5, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0 });
        }

        private void linkLabel4_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            generate_conv(new int[9] { 0, 1, 0, 1, -4, 1, 0, 1, 0 });
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            int val = (int)nudConvSize.Value;
            numud_list.ForEach(delegate (NumericUpDown numbUD) {
                this.panel1.Controls.Remove(numbUD);
            });
            numud_list.Clear();

            for (int i = 0; i < val; i++)
            {
                for (int j = 0; j < val; j++)
                {
                    NumericUpDown numbUD = new NumericUpDown();
                    numbUD.Top = i * 20 + 25;
                    numbUD.Width = 45;
                    numbUD.Minimum = -numbUD.Maximum;
                    numbUD.Left = (numbUD.Width + 20) * j;
                    if (i == (val / 2) && j == (val / 2) && val % 2 == 1) numbUD.Value = 1;
                    numud_list.Add(numbUD);
                    this.panel1.Controls.Add(numbUD);
                }
            }
        }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            cache_size = default_cach = (int)numericUpDown1.Value;
        }
    }
    /// <summary>
    /// Класс разблокированного изображения с быстрым доступом к пикселям
    /// </summary>
    public class ImagePix : IDisposable
    {
        public Bitmap bitmap { get; private set; }
        public Int32[] bits { get; private set; }
        public int height { get; private set; }
        public int width { get; private set; }
        public bool disposed { get; private set; }
        protected GCHandle bits_handle { get; private set; }

        public ImagePix(int _width, int _height)
        {
            width = _width;
            height = _height;
            bits = new Int32[width * height];
            bits_handle = GCHandle.Alloc(bits, GCHandleType.Pinned);
            bitmap = new Bitmap(width, height, width * 4, PixelFormat.Format32bppPArgb, bits_handle.AddrOfPinnedObject());
        }
        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            bitmap.Dispose();
            bits_handle.Free();
        }
        public void SetPixel(int x, int y, Color color)
        {
            bits[x + (y * width)] = color.ToArgb();

        }
        public Color GetPixel(int x, int y)
        {
            return Color.FromArgb(bits[x + (y * width)]);
        }
    }
}
