package noveltydetection.HeCluS;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.EventHandler;
import java.io.File;
import java.io.IOException;

import javax.swing.*;

public class MINASFrame extends JFrame{
	private Container c;
    private JPanel panel;
    private JButton btnExecute;
    private JButton btnFileOn;
    private JButton btnFileOff;
    private JButton btnFileOut;    
    private JButton btnCancelar;
    private JTextField txtOfflineFile;
    private JTextField txtOnlineFile;
    private JTextField txtOutDirectory;
    private JComboBox cbxAlgOffline;
    private JComboBox cbxAlgOnline;
	private JComboBox cbxEvaluation; 
	private JTextField txtNumClusters;
	private JTextField txtThresholdNovelty;
	private JTextField txtWindowSize; 
	private JTextField txtNumMinExDN;
    
	public MINASFrame (){   
		createContainer();
        this.repaint();
        this.show();
    }

	public void createContainer() {
	    FlowLayout layout;
	    
		c = getContentPane(); 
        layout = new FlowLayout();        
        panel = new JPanel();
		
        //********File Offline
		JLabel lblOfflineFile = new JLabel();
		lblOfflineFile.setText("Offline File (training): ");
		panel.add(lblOfflineFile);
		
		txtOfflineFile = new JTextField();
		txtOfflineFile.setPreferredSize(new Dimension(200,20));
		panel.add(txtOfflineFile);
		
		btnFileOff = new JButton();
		btnFileOff.setText("...");
		buttonOpenOffEvent hOpenOff = new buttonOpenOffEvent();
        btnFileOff.addActionListener(hOpenOff);
		panel.add(btnFileOff);		
		
        //********File Online
		JLabel lblOnlineFile = new JLabel();
		lblOnlineFile.setText("Online File (stream): ");
		panel.add(lblOnlineFile);
		
		txtOnlineFile = new JTextField();
		txtOnlineFile.setPreferredSize(new Dimension(200,20));
		panel.add(txtOnlineFile);
		
		btnFileOn = new JButton();
		btnFileOn.setText("...");
		buttonOpenOnEvent hOpenOn = new buttonOpenOnEvent();
        btnFileOn.addActionListener(hOpenOn);
		panel.add(btnFileOn);		

		
		//********Output Directory
		JLabel lblOutDirectory = new JLabel();
		lblOutDirectory.setText("Output Filename: ");
		panel.add(lblOutDirectory);
				
		txtOutDirectory = new JTextField();
		txtOutDirectory.setPreferredSize(new Dimension(250,20));
		panel.add(txtOutDirectory);
		
		btnFileOut = new JButton();
		btnFileOut.setText("...");
		buttonOpenOutEvent hOpenOut = new buttonOpenOutEvent();
        btnFileOut.addActionListener(hOpenOut);
		panel.add(btnFileOut);	
		
		//******************Alg Offline
		JLabel lblAlgOfflinePhase = new JLabel();
		lblAlgOfflinePhase.setText("Algorithm for the offline phase: ");
		panel.add(lblAlgOfflinePhase);
		
		String algOffline[] = {"clustream", "kmeans","clustream+kMeans"};
		cbxAlgOffline = new JComboBox(algOffline);
		panel.add(cbxAlgOffline);
		
		//*********************Alg Online
		JLabel lblAlgOnlinePhase = new JLabel();
		lblAlgOnlinePhase.setText("Algorithm for the online phase: ");
		panel.add(lblAlgOnlinePhase);
		
		String algOnline[] = {"clustream", "Kmeans"};
		cbxAlgOnline = new JComboBox(algOnline);
		cbxAlgOnline.setPreferredSize(new Dimension(150,20));
		panel.add(cbxAlgOnline);
		
		//********************* Number of Micro-Clusters
		JLabel lblNumClusters = new JLabel();
		lblNumClusters.setText("Number of micro-clusters: ");
		panel.add(lblNumClusters);
		
		txtNumClusters = new JTextField();
		txtNumClusters.setPreferredSize(new Dimension(200,20));
		txtNumClusters.setText("100");
		panel.add(txtNumClusters);
		
		//********************* Threshold		
		JLabel lblThreshold = new JLabel();
		lblThreshold.setText("Threshold for novelty detection: ");
		panel.add(lblThreshold);
		
		txtThresholdNovelty = new JTextField();
		txtThresholdNovelty.setPreferredSize(new Dimension(200,20));
		txtThresholdNovelty.setText("1.1");
		panel.add(txtThresholdNovelty);
		
		//********************* Window Size	
		JLabel lblWindowSize = new JLabel();
		lblWindowSize.setText("Window size (forget oudated data): ");
		panel.add(lblWindowSize);
		
		txtWindowSize = new JTextField();
		txtWindowSize.setPreferredSize(new Dimension(200,20));
		txtWindowSize.setText("4000");
		panel.add(txtWindowSize);
		
		//********************* minimum number to execute the novelty detection procedure	
		JLabel lblNumMinExDN = new JLabel();
		lblNumMinExDN.setText("Number of ex. to execute the ND: ");
		panel.add(lblNumMinExDN);

		txtNumMinExDN = new JTextField();
		txtNumMinExDN.setPreferredSize(new Dimension(200,20));
		txtNumMinExDN.setText("2000");
		panel.add(txtNumMinExDN);

		//********************* Evaluation
		JLabel lblTypeEvaluation = new JLabel();
		lblTypeEvaluation.setText("Methodology used in the Evaluation: ");
		panel.add(lblTypeEvaluation);
		
		String evaluation[] = {"Proposed Methodology","Literature Methodology"};
		cbxEvaluation = new JComboBox(evaluation);
		cbxEvaluation.setPreferredSize(new Dimension(200,20));
		panel.add(cbxEvaluation);
		
		//********************* Buttons
		btnExecute = new JButton();
        btnExecute.setText("Execute");
		buttonExecutionEvent hExec = new buttonExecutionEvent();
        btnExecute.addActionListener(hExec);
        panel.add(btnExecute);
                
        btnCancelar = new JButton();
        btnCancelar.setText("Cancelar");
        buttonCloseEvent hClose = new buttonCloseEvent();
        btnCancelar.addActionListener(hClose);
        panel.add(btnCancelar);
                        
        c.add(panel);       
        this.setLocation(50,50);
        this.setSize(new Dimension(450,400));	
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	public void executeMINAS() throws IOException, NumberFormatException{
		String algOff, algOn, outDirectory, filenameOff,filenameOnl;
		double threshold;
		int windowSize, numMicroClusters, numMinExDN;
		boolean flagMicro;
		if (cbxAlgOffline.getSelectedItem().toString().equals("clustream+kMeans")==true){
			algOff = "clustream";
			flagMicro = false;
		}
		else{
			algOff = cbxAlgOffline.getSelectedItem().toString();
			flagMicro = true;
		}
		algOn =cbxAlgOnline.getSelectedItem().toString();
		threshold = Double.parseDouble(txtThresholdNovelty.getText()); 
		windowSize = Integer.parseInt(txtWindowSize.getText());
		numMicroClusters = Integer.parseInt(txtNumClusters.getText());
		numMinExDN = Integer.parseInt(txtNumMinExDN.getText());
		outDirectory = txtOutDirectory.getText();
		filenameOff = txtOfflineFile.getText();
		filenameOnl = txtOnlineFile.getText();
//		MINAS m = new MINAS(filenameOff, filenameOnl,outDirectory, "clustream", "denstream", "clustree",numMicroClusters, threshold, numMinExDN, windowSize,cbxEvaluation.getSelectedIndex(),flagMicro,0,0,0);
//    	m.execution();		
	}
	
	//class for button event treatment
	class buttonExecutionEvent implements ActionListener{	
		@Override
		public void actionPerformed(ActionEvent arg0) {
			// TODO Auto-generated method stub		
			try {
				executeMINAS();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	class buttonOpenOffEvent implements ActionListener{
		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			String filepath = openFile(1);
			txtOfflineFile.setText(filepath);
		}		
	}
	class buttonOpenOnEvent implements ActionListener{
		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			String filepath = openFile(1);
			txtOnlineFile.setText(filepath);
		}	
	}
	class buttonOpenOutEvent implements ActionListener{
		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			String filepath = openFile(0);
			txtOutDirectory.setText(filepath);
			
		}		
	}
	class buttonCloseEvent implements ActionListener{
		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			close();
		}
	}
	
	private String openFile(int flg){
		JFileChooser OpenFrame = new JFileChooser();
		File file = new File("");
		String filepath = "";
	
		OpenFrame.setDialogTitle("Open File");
		if (flg == 0)
			OpenFrame.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		else
			OpenFrame.setFileSelectionMode(JFileChooser.FILES_ONLY);
		int result = OpenFrame.showOpenDialog(this);
		if (result == JFileChooser.APPROVE_OPTION) {
			file = OpenFrame.getSelectedFile();
			filepath = file.getPath();
		}
		return filepath;
	}
	
	private void close(){
		this.dispose();
	}
	public static void main(String args[]){
		MINASFrame f = new MINASFrame();
	}
}
