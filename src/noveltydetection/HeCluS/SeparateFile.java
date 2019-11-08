package noveltydetection.HeCluS;

import java.io.*;
import java.util.*;

public class SeparateFile {
	
//********************************************************************************
//**************************SepareteLabeleFiel***********************************
//********************************************************************************
	//receive a labeled file and create a file for each problem class
	public ArrayList<String> separateLabeledFile(String filename) throws IOException{
		BufferedReader fileIn = new BufferedReader(new FileReader(new File(filename)));
		String line;
		ArrayList<String> classes = new ArrayList<String>();
		ArrayList<FileWriter> fileOut = new ArrayList<FileWriter>();
		String classe;
		
		//read each line from the file		
		while ((line = fileIn.readLine()) != null){
			String[] data = line.split(",");
			classe = data[data.length-1];
			// create a file for each training class 
			int pos = classes.indexOf(classe);
			
			if (pos == -1){
				classes.add(classe);
				fileOut.add(new FileWriter(filename+classe));
				pos = fileOut.size()-1;
			}			
			fileOut.get(pos).write(line + "\n");
		}
		
		//close files
		fileIn.close();
		for (int i=0; i< fileOut.size(); i++)
			fileOut.get(i).close();
		
		ArrayList<String> vetClasses = new ArrayList<String> ();
		for (int i=0; i<classes.size();i++)
			vetClasses.add(classes.get(i));
		
		return vetClasses; 
	}
		
}
