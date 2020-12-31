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



namespace cuda_convolution_UI
{
    public partial class Form1 : Form
    {
        [DllImport("cudaFanc64.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern int getCudaDev();
        [DllImport("cudaFanc64.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr calcConvolutionCuda(int N, int M, IntPtr img, IntPtr conv, int cN);

        private int image_width;
        private int image_height;
        private List<NumericUpDown> numud_list = new List<NumericUpDown>();
        private const int default_cach = 128;
        private int cache_size = default_cach;
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

                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    pbOriginal.ImageLocation = dialog.FileName;
                }
            }
            catch (Exception)
            {
                MessageBox.Show("Not load image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public async void apply_conv_matrix(int[] conv_array, Bitmap img)
        {
            Stopwatch stopWatch2 = new Stopwatch();
            stopWatch2.Start();
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

                            IntPtr c = calcConvolutionCuda(cache_size, cache_size, pointer_img, pointer_conv, (int)Math.Sqrt(conv_array.Length));
                            int[] res = new int[cache_size * cache_size];
                            Marshal.Copy(c, res, 0, cache_size * cache_size);
                            for (int i = (new_width - 1) / cache_size * cache_size; i < new_width; i++)
                            {
                                for (int j = (new_height - 1) / cache_size * cache_size; j < new_height; j++)
                                {
                                    int r = res[i % cache_size * cache_size + (j % cache_size)];
                                    Color cl = img.GetPixel(i, j);
                                    if (chl == 0) img.SetPixel(i, j, Color.FromArgb(r, cl.G, cl.B));
                                    else if (chl == 1) img.SetPixel(i, j, Color.FromArgb(cl.R, r, cl.B));
                                    else img.SetPixel(i, j, Color.FromArgb(cl.R, cl.G, r));
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
        }

        private void pbOriginal_LoadCompleted(object sender, AsyncCompletedEventArgs e)
        {
            btnAply.Enabled = true;
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            lblImgWidth.Text = "width: " + pbOriginal.Image.Width + " height: " + pbOriginal.Image.Height;
            image_width = pbOriginal.Image.Width;
            image_height = pbOriginal.Image.Height;
            if (image_height < cache_size || image_width < cache_size)
            {
                cache_size = Math.Min(image_height, image_width);
            }
            else if (cache_size != default_cach) cache_size = default_cach;
            stFooterChild2.Text = "satus: img load - OK";
            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            stFooterChild3.Text = "time load img: " + (ts.Milliseconds).ToString() + "ms";
            
            //int[] conv_array = new int[9] { 1, 0, -1, 2, 0, -2, 1, 0, -1 };                                                     // Прикольный фильтер Собеля
            //int[] conv_array = new int[9] { -2, -1, 0, -1, 1, 1, 0, 1, 2 };                                                     // Тиснение
            //int[] conv_array = new int[25] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 };       // пацанское размытие
            //int[] conv_array = new int[25] { 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 5, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0 };   // резкость
            //int[] conv_array = new int[9] { 0, 1, 0, 1, -4, 1, 0, 1, 0 };                                                       // Хорошо выраженные края

            Bitmap img = new Bitmap(pbOriginal.Image);
            //apply_conv_matrix(conv_array, img);
            pbResault.Image = img;
        }

        private void Form1_ClientSizeChanged(object sender, EventArgs e)
        {
            pbOriginal.Width = this.Width / 2 - 40;
            pbOriginal.Height = this.Height - 120 - panel1.Height;
            lblLoading.Left = pbResault.Left = this.Width / 2 + 20;
            lblLoading.Width =  pbResault.Width = this.Width / 2 - 40;
            pbResault.Height = this.Height - 120 - panel1.Height;
            
            btnAply.Left = this.Width / 2 - 27;
            btnAply.Top = pbOriginal.Height / 2 + pbOriginal.Top - (btnAply.Height / 2);
            panel1.Top = this.Height - panel1.Height - 70;
            lblLoading.Top = pbOriginal.Height / 2 + pbOriginal.Top - (lblLoading.Height / 2);
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
            Bitmap img = new Bitmap(pbOriginal.Image);
            apply_conv_matrix(conv_matrix, img);
            pbResault.Image = img;
        }
    }
}
