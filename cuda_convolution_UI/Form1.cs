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

namespace cuda_convolution_UI
{
    public partial class Form1 : Form
    {
        [DllImport("cudaFanc64.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern int getCudaDev();
        public Form1()
        {
            InitializeComponent();
            stFooterChild1.Text = "Cuda devices: " + getCudaDev().ToString();
            if (getCudaDev() < 1)
            {
                MessageBox.Show("Not CUDA supporting", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
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

        private void pbOriginal_LoadCompleted(object sender, AsyncCompletedEventArgs e)
        {
            lblImgWidth.Text = "width: " + pbOriginal.Image.Width + " height: " + pbOriginal.Image.Height;
        }
    }
}
