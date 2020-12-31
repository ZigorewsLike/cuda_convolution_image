
namespace cuda_convolution_UI
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.stFooter = new System.Windows.Forms.StatusStrip();
            this.btnU = new System.Windows.Forms.Button();
            this.pbOriginal = new System.Windows.Forms.PictureBox();
            this.pbResault = new System.Windows.Forms.PictureBox();
            this.lblImgWidth = new System.Windows.Forms.Label();
            this.stFooterChild1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.stFooterChild2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.stFooterChild3 = new System.Windows.Forms.ToolStripStatusLabel();
            this.stFooterChild4 = new System.Windows.Forms.ToolStripStatusLabel();
            this.stFooterChild5 = new System.Windows.Forms.ToolStripStatusLabel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.nudConvSize = new System.Windows.Forms.NumericUpDown();
            this.lblform = new System.Windows.Forms.Label();
            this.btnAply = new System.Windows.Forms.Button();
            this.lblLoading = new System.Windows.Forms.Label();
            this.stFooter.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbOriginal)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbResault)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.nudConvSize)).BeginInit();
            this.SuspendLayout();
            // 
            // stFooter
            // 
            this.stFooter.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.stFooterChild1,
            this.stFooterChild2,
            this.stFooterChild3,
            this.stFooterChild4,
            this.stFooterChild5});
            this.stFooter.Location = new System.Drawing.Point(0, 757);
            this.stFooter.Name = "stFooter";
            this.stFooter.Size = new System.Drawing.Size(962, 22);
            this.stFooter.SizingGrip = false;
            this.stFooter.TabIndex = 0;
            this.stFooter.Text = "statusStrip1";
            // 
            // btnU
            // 
            this.btnU.Location = new System.Drawing.Point(0, 0);
            this.btnU.Name = "btnU";
            this.btnU.Size = new System.Drawing.Size(154, 29);
            this.btnU.TabIndex = 1;
            this.btnU.Text = "Загрузить пикчу";
            this.btnU.UseVisualStyleBackColor = true;
            this.btnU.Click += new System.EventHandler(this.btnU_Click);
            // 
            // pbOriginal
            // 
            this.pbOriginal.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbOriginal.Location = new System.Drawing.Point(1, 35);
            this.pbOriginal.Name = "pbOriginal";
            this.pbOriginal.Size = new System.Drawing.Size(460, 422);
            this.pbOriginal.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbOriginal.TabIndex = 2;
            this.pbOriginal.TabStop = false;
            this.pbOriginal.LoadCompleted += new System.ComponentModel.AsyncCompletedEventHandler(this.pbOriginal_LoadCompleted);
            // 
            // pbResault
            // 
            this.pbResault.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbResault.Location = new System.Drawing.Point(502, 35);
            this.pbResault.Name = "pbResault";
            this.pbResault.Size = new System.Drawing.Size(460, 422);
            this.pbResault.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbResault.TabIndex = 3;
            this.pbResault.TabStop = false;
            this.pbResault.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pbResault_MouseMove);
            // 
            // lblImgWidth
            // 
            this.lblImgWidth.AutoSize = true;
            this.lblImgWidth.Location = new System.Drawing.Point(160, 7);
            this.lblImgWidth.Name = "lblImgWidth";
            this.lblImgWidth.Size = new System.Drawing.Size(88, 13);
            this.lblImgWidth.TabIndex = 4;
            this.lblImgWidth.Text = "width: 0 height: 0";
            // 
            // stFooterChild1
            // 
            this.stFooterChild1.Name = "stFooterChild1";
            this.stFooterChild1.Size = new System.Drawing.Size(89, 17);
            this.stFooterChild1.Text = "Cuda devices: 0";
            // 
            // stFooterChild2
            // 
            this.stFooterChild2.Name = "stFooterChild2";
            this.stFooterChild2.Size = new System.Drawing.Size(66, 17);
            this.stFooterChild2.Text = "status: chill";
            // 
            // stFooterChild3
            // 
            this.stFooterChild3.Name = "stFooterChild3";
            this.stFooterChild3.Size = new System.Drawing.Size(88, 17);
            this.stFooterChild3.Text = "time load: 0 ms";
            // 
            // stFooterChild4
            // 
            this.stFooterChild4.Name = "stFooterChild4";
            this.stFooterChild4.Size = new System.Drawing.Size(82, 17);
            this.stFooterChild4.Text = "time edit 0 ms";
            // 
            // stFooterChild5
            // 
            this.stFooterChild5.Name = "stFooterChild5";
            this.stFooterChild5.Size = new System.Drawing.Size(74, 17);
            this.stFooterChild5.Text = "mouse pos 0";
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.nudConvSize);
            this.panel1.Controls.Add(this.lblform);
            this.panel1.Location = new System.Drawing.Point(12, 463);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(637, 268);
            this.panel1.TabIndex = 8;
            // 
            // nudConvSize
            // 
            this.nudConvSize.Location = new System.Drawing.Point(169, 3);
            this.nudConvSize.Maximum = new decimal(new int[] {
            9,
            0,
            0,
            0});
            this.nudConvSize.Minimum = new decimal(new int[] {
            3,
            0,
            0,
            0});
            this.nudConvSize.Name = "nudConvSize";
            this.nudConvSize.Size = new System.Drawing.Size(67, 20);
            this.nudConvSize.TabIndex = 9;
            this.nudConvSize.Value = new decimal(new int[] {
            3,
            0,
            0,
            0});
            this.nudConvSize.ValueChanged += new System.EventHandler(this.nudConvSize_ValueChanged);
            // 
            // lblform
            // 
            this.lblform.AutoSize = true;
            this.lblform.Location = new System.Drawing.Point(0, 5);
            this.lblform.Name = "lblform";
            this.lblform.Size = new System.Drawing.Size(167, 13);
            this.lblform.TabIndex = 10;
            this.lblform.Text = "Размерность матрицы свёртки";
            // 
            // btnAply
            // 
            this.btnAply.Enabled = false;
            this.btnAply.Location = new System.Drawing.Point(463, 209);
            this.btnAply.Name = "btnAply";
            this.btnAply.Size = new System.Drawing.Size(37, 35);
            this.btnAply.TabIndex = 11;
            this.btnAply.Text = ">";
            this.btnAply.UseVisualStyleBackColor = true;
            this.btnAply.Click += new System.EventHandler(this.button1_Click);
            // 
            // lblLoading
            // 
            this.lblLoading.BackColor = System.Drawing.Color.White;
            this.lblLoading.Font = new System.Drawing.Font("Microsoft Sans Serif", 32F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.lblLoading.Location = new System.Drawing.Point(503, 199);
            this.lblLoading.Name = "lblLoading";
            this.lblLoading.Size = new System.Drawing.Size(459, 52);
            this.lblLoading.TabIndex = 12;
            this.lblLoading.Text = "ОБРАБОТКА ...";
            this.lblLoading.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.lblLoading.Visible = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(962, 779);
            this.Controls.Add(this.lblLoading);
            this.Controls.Add(this.btnAply);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.lblImgWidth);
            this.Controls.Add(this.pbResault);
            this.Controls.Add(this.pbOriginal);
            this.Controls.Add(this.btnU);
            this.Controls.Add(this.stFooter);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Form1";
            this.ClientSizeChanged += new System.EventHandler(this.Form1_ClientSizeChanged);
            this.stFooter.ResumeLayout(false);
            this.stFooter.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbOriginal)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbResault)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.nudConvSize)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.StatusStrip stFooter;
        private System.Windows.Forms.Button btnU;
        private System.Windows.Forms.PictureBox pbOriginal;
        private System.Windows.Forms.PictureBox pbResault;
        private System.Windows.Forms.ToolStripStatusLabel stFooterChild1;
        private System.Windows.Forms.ToolStripStatusLabel stFooterChild2;
        private System.Windows.Forms.ToolStripStatusLabel stFooterChild3;
        private System.Windows.Forms.ToolStripStatusLabel stFooterChild4;
        private System.Windows.Forms.ToolStripStatusLabel stFooterChild5;
        private System.Windows.Forms.Label lblImgWidth;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.NumericUpDown nudConvSize;
        private System.Windows.Forms.Label lblform;
        private System.Windows.Forms.Button btnAply;
        private System.Windows.Forms.Label lblLoading;
    }
}

