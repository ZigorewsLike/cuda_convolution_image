
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
            this.stFooter.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbOriginal)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbResault)).BeginInit();
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
            this.stFooter.Location = new System.Drawing.Point(0, 496);
            this.stFooter.Name = "stFooter";
            this.stFooter.Size = new System.Drawing.Size(954, 22);
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
            // 
            // pbOriginal
            // 
            this.pbOriginal.Location = new System.Drawing.Point(2, 53);
            this.pbOriginal.Name = "pbOriginal";
            this.pbOriginal.Size = new System.Drawing.Size(460, 422);
            this.pbOriginal.TabIndex = 2;
            this.pbOriginal.TabStop = false;
            // 
            // pbResault
            // 
            this.pbResault.Location = new System.Drawing.Point(485, 53);
            this.pbResault.Name = "pbResault";
            this.pbResault.Size = new System.Drawing.Size(460, 422);
            this.pbResault.TabIndex = 3;
            this.pbResault.TabStop = false;
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
            this.stFooterChild1.Size = new System.Drawing.Size(118, 17);
            this.stFooterChild1.Text = "toolStripStatusLabel1";
            // 
            // stFooterChild2
            // 
            this.stFooterChild2.Name = "stFooterChild2";
            this.stFooterChild2.Size = new System.Drawing.Size(118, 17);
            this.stFooterChild2.Text = "toolStripStatusLabel1";
            // 
            // stFooterChild3
            // 
            this.stFooterChild3.Name = "stFooterChild3";
            this.stFooterChild3.Size = new System.Drawing.Size(118, 17);
            this.stFooterChild3.Text = "toolStripStatusLabel1";
            // 
            // stFooterChild4
            // 
            this.stFooterChild4.Name = "stFooterChild4";
            this.stFooterChild4.Size = new System.Drawing.Size(118, 17);
            this.stFooterChild4.Text = "toolStripStatusLabel1";
            // 
            // stFooterChild5
            // 
            this.stFooterChild5.Name = "stFooterChild5";
            this.stFooterChild5.Size = new System.Drawing.Size(118, 17);
            this.stFooterChild5.Text = "toolStripStatusLabel1";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(954, 518);
            this.Controls.Add(this.lblImgWidth);
            this.Controls.Add(this.pbResault);
            this.Controls.Add(this.pbOriginal);
            this.Controls.Add(this.btnU);
            this.Controls.Add(this.stFooter);
            this.Name = "Form1";
            this.Text = "Form1";
            this.stFooter.ResumeLayout(false);
            this.stFooter.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbOriginal)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbResault)).EndInit();
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
    }
}

