package noveltydetection.HeCluS;

import java.io.*;
import java.util.*;

public class Evaluator {	
	ArrayList<String> trainingClasses = new ArrayList<String>();
	ArrayList<String> classNames = new ArrayList<String>();
	ArrayList<String> columnNames = new ArrayList<String>();	
	Hashtable confusionMatrix = new Hashtable();
	
	List<Float> measure1 = new ArrayList<Float>();
	List<Float> measure2 = new ArrayList<Float>();
	List<Float> measure3 = new ArrayList<Float>();
	List<Float> timestamps = new ArrayList<Float>();
	List<Integer> timestampNovidade = new ArrayList<Integer>();
	List<Integer> timestampExtension = new ArrayList<Integer>();
	
	int nroElemConfusionMatrix;
	int evaluationType;

//*********************************************************************************************
//***************************** Evaluator *****************************************************
//*********************************************************************************************		
	public Evaluator(int flag){
		evaluationType = flag;
	}
	
//*********************************************************************************************
//***************************** printConfusionMatrix ******************************************
//*********************************************************************************************		
	public void printConfusionMatrix(String description,FileWriter fileOut, Hashtable matrix) throws IOException{
		System.out.println("\n" +description + ": \n");
		fileOut.write("\n" + description + ": \n");
		for (int i=0; i< columnNames.size(); i++){
			System.out.print("\t" + columnNames.get(i));
			fileOut.write("," + columnNames.get(i));
		}
		System.out.print("\n ------------------------------------------------------------------------------------------------");
		
		for (int i=0; i<matrix.size();i++){
			System.out.print("\n" + classNames.get(i) );
			fileOut.write("\n" + classNames.get(i));
			for (int j=0; j< ((Hashtable)matrix.get(classNames.get(i))).size(); j++){
				System.out.print("\t" + ((Hashtable)matrix.get(classNames.get(i))).get(columnNames.get(j)));
				fileOut.write("," + ((Hashtable)matrix.get(classNames.get(i))).get(columnNames.get(j)));
			}
			System.out.print("\n");
		}
		System.out.println("\n \n");
	}

//*********************************************************************************************
//***************************** initializeConfusionMatrix *************************************
//*********************************************************************************************	
	public void initializeConfusionMatrix(String filename)throws IOException{
		//open file: stores the number of problem classes and their labels
		BufferedReader fileIn = new BufferedReader(new FileReader(new File(filename+"Classes")));
		String line;
				
		while ((line = fileIn.readLine()) != null){
			trainingClasses.add("C "+ line);
			classNames.add(trainingClasses.get(trainingClasses.size()-1));
			columnNames.add(trainingClasses.get(trainingClasses.size()-1));
		}
		Collections.sort(trainingClasses);
		Collections.sort(classNames);
		Collections.sort(columnNames);	
		columnNames.add("não sei");
		
		//create the hash table: stores the confusion matrix
		for (int i=0; i<classNames.size();i++){
			confusionMatrix.put(classNames.get(i), new Hashtable());
			for (int j=0; j<columnNames.size(); j++){	
				((Hashtable)confusionMatrix.get(classNames.get(i))).put(columnNames.get(j), 0);
			}
		}	
		nroElemConfusionMatrix = 0;
		fileIn.close();
		
		measure1.add(0.0f);
		measure3.add(0.0f);		
		timestamps.add((float)(nroElemConfusionMatrix/1000));
		
		if (evaluationType !=0 )
			measure2.add(0.0f);
		else
			measure2.add(100.0f);
	}
	
//*********************************************************************************************
//***************************** addElementConfusionMatrix *************************************
//*********************************************************************************************	
	public int addElementConfusionMatrix(String infoExampleClass) throws IOException{														
		String trueClass="", MINASClass="";						
		int value;

		//read the line that contains information about the true class of the example and the MINAS class
		String[] data = infoExampleClass.split("\t");
		trueClass = "C" + data[1].split(":")[1];			
		MINASClass = data[2].split(":")[1].trim();

		//examples belonging to an extension of a known concept will be computed as belonging to the concept 
		if (MINASClass.split(" ")[0].equals("ExtCon")){
			MINASClass = MINASClass.replace("ExtCon", "C");
		}
		//examples belonging to an extension of a novelty will be computed as belonging to the novelty
		else if (MINASClass.split(" ")[0].equals("ExtNov")){
			MINASClass = MINASClass.replace("ExtNov", "N");
		}
		//if new problem class appears, add a line in the confusion matrix
		if (classNames.indexOf(trueClass)==-1){
			classNames.add(trueClass); 
			confusionMatrix.put(trueClass, new Hashtable());
			for (int j=0; j<columnNames.size(); j++){	
				((Hashtable)confusionMatrix.get(trueClass)).put(columnNames.get(j), 0);
			}
		}
		value = Integer.parseInt((((Hashtable)confusionMatrix.get(trueClass)).get(MINASClass)).toString());
		//update the confusion matrix in the corresponding cell
		((Hashtable)confusionMatrix.get(trueClass)).put(MINASClass,value+1); 

		nroElemConfusionMatrix++; //update the count of read examples
		//calculate evaluation measures each 1000 timestamps
		if (nroElemConfusionMatrix%1000 == 0){
			calculateEvaluationMeasures();
		}
		return 0;
	}
	
//*********************************************************************************************
//***************************** calculateEvaluationMeasures ***********************************
//*********************************************************************************************		
	public void calculateEvaluationMeasures(){	
		//calculate evaluation measures using the current confusion matrix 
		if (evaluationType ==0 ){// Proposed Methodology + Combined Error
			ArrayList<Integer> numExNoUnkPerClass = new ArrayList<Integer>();
			float results[] = calculateUnkRate(numExNoUnkPerClass);
			float combinedError = calculateCombinedError(numExNoUnkPerClass);
			float FMacro = calculateEvaluationMeasureFMacro();
			measure1.add(results[0]);//Unk				
			measure2.add(FMacro);//FMacro
			measure3.add(combinedError);//CER
			timestamps.add((float)(nroElemConfusionMatrix/1000));
		}
		else if (evaluationType ==1 ){ // Methodology from the literature
			float results[] = calculateEvaluationMeasureLiterature();
			measure1.add(results[0]);//FNew
			measure2.add(results[1]);//MNew
			measure3.add(results[2]);//Err
			timestamps.add((float)(nroElemConfusionMatrix/1000));
		}
		else if (evaluationType ==2 ){//Proposed Methodology + Error Rate
			ArrayList<Integer> numExNoUnkPerClass = new ArrayList<Integer>();
			float res[] = calculateUnkRate(numExNoUnkPerClass);
			float errorRate = calculateErrorRate(numExNoUnkPerClass);
			float FMacro = calculateEvaluationMeasureFMacro();
			measure1.add(res[0]);//Unk				
			measure2.add(FMacro);//FMacro
			measure3.add(errorRate);//ErrorRate
			timestamps.add((float)(nroElemConfusionMatrix/1000));				
		}
	}
//*********************************************************************************************
//***************************** addColumnConfusionMatrix **************************************
//*********************************************************************************************	
	public void addColumnConfusionMatrix(String info, Hashtable tmpMatrix){
		String classe;
		if (info.contains("ThinkingNov")){//information contains the word thinkingNov --> then new novelty pattern
			timestampNovidade.add(nroElemConfusionMatrix/1000);
			String nomeNovidade = "N " + info.split(" ")[2];											
			//insert a new column in the confusion matrix
			columnNames.add(columnNames.get(columnNames.size()-1));
			columnNames.set(columnNames.size()-2, nomeNovidade);
			for (int i=0; i<classNames.size();i++){
				((Hashtable)confusionMatrix.get(classNames.get(i))).put(nomeNovidade, 0);
			}
			classe = nomeNovidade;			
		} 
		else{ // it is an extension
			//arraylist contains the timestamp where new extensions were detected
			timestampExtension.add(nroElemConfusionMatrix/1000); 
			classe = info.split(":")[1].split(" -")[0].trim();
		}
		
		//update the confusion matrix: remove the examples from the unknown column and put them in the corresponding class 
		Integer valor, valorUnk, valorC;
		for (int i=0; i< classNames.size();i++){
			//valor correspondente ã classe na matriz de confusao parcial
			if (tmpMatrix.get(classNames.get(i))!=null){
				valor = ((Integer)tmpMatrix.get(classNames.get(i))).intValue();
				//remove from unknown
				valorUnk = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get("não sei").toString());
				valorUnk = valorUnk - valor;
				((Hashtable)confusionMatrix.get(classNames.get(i))).put("não sei", valorUnk);
				//put in the corresponding class
				valorC = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get(classe).toString());
				valorC = valorC + valor;				
				((Hashtable)confusionMatrix.get(classNames.get(i))).put(classe, valorC);					
			}
		}
	}
		
//*********************************************************************************************
//***************************** showResults ***************************************************
//*********************************************************************************************	
	public void showResults(String filename)throws IOException{
		FileWriter fileOut = new FileWriter(filename+"Matriz.csv");
		
		//***********************************************************
		// print final confusion matrix      
		//************************************************************
		printConfusionMatrix("Resultado final", fileOut,confusionMatrix);

		//*****************************************************
		// calculate the confusion matrix using percentage
		//*****************************************************	
		Hashtable matConfusaoPct = calculateMatrixPercentage();

		//*****************************************************
		// print the confusion matrix using percentage
		//*****************************************************	
		printConfusionMatrix("Impressao em forma de porcentagem", fileOut, matConfusaoPct);

		fileOut.close();

		//calculate evaluation measures in the last timestamp (after read all examples from the online phase)
		calculateEvaluationMeasureFMacro();
		
		//save thre results in a file
		saveResults (filename); 
		
		//create a graphic (using FreeChart) to show the evaluation measures
		FreeChartGraph t = new FreeChartGraph(filename);
		t.createGraphic(measure1, measure2, measure3,timestamps,timestampNovidade,timestampExtension,evaluationType);
		// show the graphic using a JFrame
//		GraphicFrame frame = new GraphicFrame(t.getGraphic());
	}
	
//***************************************************************************************************
//**************************saveResults**************************************************************	
//***************************************************************************************************	
	public void saveResults (String filename) throws IOException{
		FileWriter f = new FileWriter(new File(filename + "_measures.txt"),false);
		if (evaluationType == 0)
			f.write("Unk,FMacro,CER\n");
		else if (evaluationType == 1)
			f.write("FNew,MNew,Err\n");
		else if (evaluationType == 2)
			f.write("Unk,FMacro,ErrorRate\n");
		
		for (int i=0; i< measure1.size(); i++){
			f.write(measure1.get(i)+ ","+ measure2.get(i)+ "," + measure3.get(i));
			f.write("\n");
		}
		f.close();
		
		FileWriter f1 = new FileWriter(new File(filename+"_novelty.txt"),false);
		f1.write("Novelties\n");
		for (int i=0; i< timestampNovidade.size(); i++){
			if (i>0){
				if (Integer.parseInt(timestampNovidade.get(i)+"")!=Integer.parseInt(timestampNovidade.get(i-1)+"")){
					f1.write(timestampNovidade.get(i)+"");			
					f1.write("\n");
				}
			}
			else{
				f1.write(timestampNovidade.get(i)+"");			
				f1.write("\n");
			}
		}
		f1.close();
		
		FileWriter f2 = new FileWriter(new File("C:\\Extension.txt"),false);
		f2.write("Extensions\n");
		for (int i=0; i< timestampExtension.size(); i++){
			if (i>0){
				if (Integer.parseInt(timestampExtension.get(i)+"") != Integer.parseInt(timestampExtension.get(i-1)+"")){
					f2.write(timestampExtension.get(i)+"");			
					f2.write("\n");

				}
			}				
			else {
				f2.write(timestampExtension.get(i)+"");			
				f2.write("\n");
			}
		}
		f2.close();
	}
		
//***************************************************************************************************
//**************************calculateMatrixPercentage************************************************	
//***************************************************************************************************	
	public Hashtable calculateMatrixPercentage(){	
		int sumLine = 0;
		Hashtable matrixPercentage = new Hashtable();
		//create a hashtabel to represent the confusion matrix (using percentage)
		for (int i=0; i<classNames.size();i++){
			matrixPercentage.put(classNames.get(i), new Hashtable());
			for (int j=0; j<columnNames.size(); j++){		
				((Hashtable)matrixPercentage.get(classNames.get(i))).put(columnNames.get(j), 0);
			}
		}
		int cell;
		for (int i=0; i< classNames.size(); i++){
			sumLine = 0;
			for (int j=0; j< columnNames.size(); j++){
				sumLine += Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get(columnNames.get(j)).toString());
			}
			for (int j=0; j< columnNames.size(); j++){
				cell = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get(columnNames.get(j)).toString());
				((Hashtable)matrixPercentage.get(classNames.get(i))).put(columnNames.get(j),(double)cell/sumLine);
			}
		}
		return matrixPercentage;
	}
	
//***************************************************************************************************
//**************************** calculateEvaluationMeasureFMacro**************************************
//***************************************************************************************************
	private float calculateEvaluationMeasureFMacro(){
		float results[], FMeasureMacro;
		int numProbemClasses = classNames.size();
		
		int numFalsePositivePerClass[],numFalseNegativePerClass[],numTruePositivePerClass[];
		numFalsePositivePerClass = new int[numProbemClasses];
		numFalseNegativePerClass = new int[numProbemClasses];
		numTruePositivePerClass = new int[numProbemClasses];
		
		Hashtable<String,ArrayList <Integer>> correspProblemClasses_MINASClasses = calcularCorrespondCProbCMinas(true);
		calculateFPFN(correspProblemClasses_MINASClasses,numTruePositivePerClass,numFalsePositivePerClass,numFalseNegativePerClass);
		results = calculateFMeasure(numTruePositivePerClass, numFalsePositivePerClass, numFalseNegativePerClass);
		FMeasureMacro = results[1]*100;
		return FMeasureMacro;
	}
	
//***************************************************************************************************
//**************************** calculaTaxasUnk********************************************
//***************************************************************************************************	
	private float[] calculateUnkRate(ArrayList<Integer> numExNoUnkPerClass){	
		float unkRatePerClass[];
		int unkPerClass[],numExPerClass[];
		float meanUnkRate=0,unkRate;
		int cell,numExamplesUnk=0,numExamples=0;
		
		int numProblemClasses = classNames.size();
		unkPerClass = new int[numProblemClasses];
		numExPerClass = new int[numProblemClasses];
		unkRatePerClass = new float[numProblemClasses];
		
		//calculate the number of unknown examples and the unknown rate per class
		for (int i=0; i<numProblemClasses; i++){
			for (int j=0;j<columnNames.size(); j++){
				cell = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get(columnNames.get(j)).toString());
				if (columnNames.get(j).compareTo("não sei") == 0){
					unkPerClass[i] = cell;
				}
				numExPerClass[i] = numExPerClass[i] + cell;
			}
			
			numExNoUnkPerClass.add(numExPerClass[i] - unkPerClass[i]);
			numExamplesUnk = numExamplesUnk + unkPerClass[i];
			numExamples = numExamples + numExPerClass[i];
			if (numExPerClass[i] == 0){
				unkRatePerClass[i] =0;
			}
			else
				unkRatePerClass[i] = (float)unkPerClass[i]/numExPerClass[i];
			meanUnkRate = meanUnkRate + unkRatePerClass[i];
		}
		unkRate = (float)numExamplesUnk/numExamples;
		meanUnkRate = meanUnkRate/classNames.size();
						
		float results[] = new float[2];
		results[0] = meanUnkRate*100;
		results[1] = unkRate*100;

		return results;
	}
	
//***************************************************************************************************
//**************************** calculateUnknRate*****************************************************
//***************************************************************************************************	
	private float calculateErrorRate(ArrayList<Integer> numExNoUnkPerClass){
				
		int numFalsePositivePerClass[],numFalseNegativePerClass[],numTruePositivePerClass[], numTrueNegativePerClass[];
		int numProblemClasses = classNames.size();
		numFalsePositivePerClass = new int[numProblemClasses];
		numFalseNegativePerClass = new int[numProblemClasses];
		numTruePositivePerClass = new int[numProblemClasses];
		numTrueNegativePerClass = new int[numProblemClasses];
		
		//calculate the number of examples classified by the system, i.e, not classified as unknown
		int numExNoUnk=0;
		for (int e=0;e<numProblemClasses;e++){
			numExNoUnk = numExNoUnk + numExNoUnkPerClass.get(e);
		}
		
		//calcualte false positive and false negative
		Hashtable<String,ArrayList <Integer>> correspProblemClasses_MINASClasses = calcularCorrespondCProbCMinas(true);
		calculateFPFN(correspProblemClasses_MINASClasses,numTruePositivePerClass,numFalsePositivePerClass,numFalseNegativePerClass);
		//calcualte true negative
		for (int e=0;e<numTrueNegativePerClass.length;e++){
			numTrueNegativePerClass[e] = numExNoUnk-(numTruePositivePerClass[e]+ numFalseNegativePerClass[e]+numFalsePositivePerClass[e]);
		}		
		//calculate the error rate
		float errorRate, errorPerClass,sumErrorPerClass=0;
		for (int p=0;p<numProblemClasses;p++){
			errorPerClass = ((float)(numFalsePositivePerClass[p]+numFalseNegativePerClass[p]))/(numFalsePositivePerClass[p]+numFalseNegativePerClass[p]+numTrueNegativePerClass[p]+numTruePositivePerClass[p]);
			sumErrorPerClass+= errorPerClass;
		}
		errorRate = sumErrorPerClass/numProblemClasses;
		
		return errorRate*100;
	}
		
//***************************************************************************************************
//**************************** calculaErroCombinado********************************************
//***************************************************************************************************	
	private float calculateCombinedError(ArrayList<Integer> numExNoUnkPerClass){
		float combinedErrorPerClass[];
		float combinedError;
		int numFalsoPositivePerClass[],numFalseNegativePerClass[],numTruePositivePerClass[], numTrueNegativePerClass[];
		
		int numProblemClasses = classNames.size();
		combinedErrorPerClass = new float[numProblemClasses];
		numFalsoPositivePerClass = new int[numProblemClasses];
		numFalseNegativePerClass = new int[numProblemClasses];
		numTruePositivePerClass = new int[numProblemClasses];
		numTrueNegativePerClass = new int[numProblemClasses];
		
		//calculate the number of examples classified by the system, i.e, not classified as unknown
		int numExNoUnk=0;
		for (int e=0;e<numProblemClasses;e++){
			numExNoUnk = numExNoUnk + numExNoUnkPerClass.get(e);
		}
		
		//calcualte false positive and false negative
		Hashtable<String,ArrayList <Integer>> correspProblemClasses_MINASClasses = calcularCorrespondCProbCMinas(true);
		calculateFPFN(correspProblemClasses_MINASClasses,numTruePositivePerClass,numFalsoPositivePerClass,numFalseNegativePerClass);
		//calcula verdadeiro negativo
		for (int e=0;e<numTrueNegativePerClass.length;e++){
			numTrueNegativePerClass[e] = numExNoUnk-(numTruePositivePerClass[e]+ numFalseNegativePerClass[e]+numFalsoPositivePerClass[e]);
		}

		//calculate combined error per class
		float falsePositiveRate, falseNegativeRate,sumCombinedErrorPerClass=0;
		for (int e=0; e<combinedErrorPerClass.length;e++){
			if ((numFalsoPositivePerClass[e]+numTrueNegativePerClass[e])!=0)
				falsePositiveRate = (float)numFalsoPositivePerClass[e]/(numFalsoPositivePerClass[e]+numTrueNegativePerClass[e]);
			else
				falsePositiveRate = 0;
			if ((numFalseNegativePerClass[e]+numTruePositivePerClass[e])!=0)
				falseNegativeRate = (float)numFalseNegativePerClass[e]/(numFalseNegativePerClass[e]+numTruePositivePerClass[e]);
			else
				falseNegativeRate = 0;
			combinedErrorPerClass[e] = numExNoUnkPerClass.get(e)*(falsePositiveRate + falseNegativeRate);
			sumCombinedErrorPerClass = sumCombinedErrorPerClass + combinedErrorPerClass[e];
		}
		if (numExNoUnk == 0)
			combinedError = 0.0f;
		else combinedError = sumCombinedErrorPerClass/(2*numExNoUnk);

		return combinedError*100;
	}
	
//***************************************************************************************************
//**************************** calculateEvaluationMeasureLiterature**********************************
//***************************************************************************************************
	private float[] calculateEvaluationMeasureLiterature(){
		int FpPerClasse[], FnPerClasse[], unkPerClass[], FePerClasse[];		
		int i, cell;
		int numExPerClass[], nroTotalExSemUnkPorClasse[];	
				
		//initialize the data structures
		int numProblemClasses = classNames.size();
		FpPerClasse = new int[trainingClasses.size()];
		FePerClasse = new int[trainingClasses.size()];
		FnPerClasse = new int[numProblemClasses-trainingClasses.size()];
		unkPerClass = new int[numProblemClasses];
		numExPerClass = new int[numProblemClasses];
		nroTotalExSemUnkPorClasse = new int[numProblemClasses];
			
		//calculat the Fe and Fp per class in the known classes 
		for (i=0; i<trainingClasses.size(); i++){
			for (int j=0;j<((Hashtable)confusionMatrix.get(trainingClasses.get(i))).size(); j++){
				cell = Integer.parseInt(((Hashtable)confusionMatrix.get(trainingClasses.get(i))).get(columnNames.get(j)).toString());
				numExPerClass[i] = numExPerClass[i] + cell;
				if (trainingClasses.indexOf(columnNames.get(j))!= -1){
					if (trainingClasses.get(i).compareTo(columnNames.get(j))!=0)
						FePerClasse[i] = FePerClasse[i] + cell; 
				}
				else if (columnNames.get(j).compareTo("não sei") == 0){
					unkPerClass[i] = cell;
				}
				else {
					FpPerClasse[i] = FpPerClasse[i] + cell;
				}
				nroTotalExSemUnkPorClasse[i] = numExPerClass[i] - unkPerClass[i];
			}
		}
		
		//calculate the Fn per class in the novel classes
		for(int k=i;k<numProblemClasses;k++){
			for (int j=0;j<columnNames.size(); j++){
				cell = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(k))).get(columnNames.get(j)).toString());
				numExPerClass[k] = numExPerClass[k] + cell;
				if (classNames.indexOf(columnNames.get(j))!=-1)
					FnPerClasse[k-i] = FnPerClasse[k-i] + cell;
				if (columnNames.get(j).compareTo("não sei") == 0)
					unkPerClass[k] = cell;
			}
			nroTotalExSemUnkPorClasse[k] = numExPerClass[k] - unkPerClass[k];
		}
		
		//calculate Mnew (Fn*100/Nc)  Fnew ((Fp*100)/(N-Nc) e Err((Fp + Fn +Fe)*100/N)
		//FNew
		float FNew, Fp = 0, numExNormal=0, Fe = 0; // (N - Nc) 
		for (i=0; i<trainingClasses.size(); i++){
			//Fp = Fp + FpPorClasse[i];
			Fp = Fp + FpPerClasse[i] + unkPerClass[i];
			numExNormal = numExNormal + numExPerClass[i];			
			//NroExCNormal = NroExCNormal + nroTotalExSemUnkPorClasse[i];
			Fe = Fe + FePerClasse[i];
		}
		if (numExNormal == 0)
			FNew = 0.0f;
		else FNew = (float)(Fp*100) /numExNormal;
		
		//Mnew
		float MNew=0, Fn=0, numExNovelty=0;
		for (i=trainingClasses.size(); i<numProblemClasses; i++){
			Fn = Fn + FnPerClasse[i-trainingClasses.size()];
			//NroExNov = NroExNov + nroTotalExSemUnkPorClasse[i];
			numExNovelty = numExNovelty + numExPerClass[i];
		}	
		if (numExNovelty == 0.0)
			MNew = 0.0f;
		else MNew = (float)(Fn*100) / numExNovelty;
				
		//Err ((Fp+Fn+Fe)*100)/N
		float Err;		
		Err = (float)((Fp + Fn + Fe)*100)/(numExNormal+numExNovelty);
				
		float results[] = new float [3];
		results[0] = FNew;       
		results[1] = MNew; 
		results[2] = Err;
		return results;
	}

//***************************************************************************************************
//**************************** calcularCorrespondCMinasCProb ****************************************
//***************************************************************************************************
	private Hashtable<String,ArrayList <Integer>> calcularCorrespondCProbCMinas(boolean algSupervisionado){
		int max=0; int posMax=0,pos;
		int cell;
		
		Hashtable<String,ArrayList <Integer>> correspProblemClasses_MINASClasses = new Hashtable<String,ArrayList<Integer>>();
		//initialize the vector that contains the correspondence among problem class and novelyt patterns
		for (int i=0; i<classNames.size();i++){
			correspProblemClasses_MINASClasses.put(classNames.get(i), new ArrayList());
		}
					
		for (int i=0; i< columnNames.size(); i++){
			if (columnNames.get(i).compareTo("não sei")!=0){
				pos =trainingClasses.indexOf(columnNames.get(i)); 
				if (pos!=-1){
					correspProblemClasses_MINASClasses.get(trainingClasses.get(pos)).add(i);
				}
				else{
					posMax = 0; max = 0;
					for (int j=0;j<classNames.size();j++){
							cell = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(j))).get(columnNames.get(i)).toString());
							if (cell > max){
								max = cell;
								posMax = j;
							}
					}
					correspProblemClasses_MINASClasses.get(classNames.get(posMax)).add(i);
				}
			}
		}
				
		return correspProblemClasses_MINASClasses;
	}
	
//**********************************************************************************************
//************************* calculateFPFN ******************************************************
//**********************************************************************************************	
	private void calculateFPFN(Hashtable<String,ArrayList <Integer>> vetorCorrespondCProbCMINAS,int numTruePositivePerClass[], int numFalsePositivePerClass[], int numFalseNegativePerClass[]){
		int cell,pos;
		
		for (int i=0;i<classNames.size();i++){
			for (int j=0;j<columnNames.size();j++){
				if (columnNames.get(j).compareTo("não sei")!=0){
					cell = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get(columnNames.get(j)).toString());
					//if the current column corresponds to the current class														
					if (belongs(j,vetorCorrespondCProbCMINAS.get(classNames.get(i)))== 1)
						numTruePositivePerClass[i] = numTruePositivePerClass[i]+cell;
					else
						numFalseNegativePerClass[i] = numFalseNegativePerClass[i]+cell;
				}
			}
		}
		
		String key;
		for(int j=0;j<columnNames.size();j++){
			if (columnNames.get(j).compareTo("não sei")!=0){
				key = search(j,vetorCorrespondCProbCMINAS);
				pos = classNames.indexOf(key);
				for (int i=0;i<classNames.size();i++){
					cell = Integer.parseInt(((Hashtable)confusionMatrix.get(classNames.get(i))).get(columnNames.get(j)).toString());
					if (i != pos){
						numFalsePositivePerClass[pos] = numFalsePositivePerClass[pos] + cell;
					}
				}
			}
		}			
	}
	
//**********************************************************************************************
//****************************** belongs *******************************************************	
//**********************************************************************************************	
	private int belongs (int elem, ArrayList<Integer> vet){
		for (int i=0; i<vet.size();i++)
			if (elem == (Integer)vet.get(i))
				return 1;
		return 0;
	}
	
//**********************************************************************************************
//******************************* search ******************************************************	
//**********************************************************************************************	
	private String search(int element, Hashtable<String,ArrayList<Integer>> matrix){
		ArrayList<Integer> line;
		
		for (int i=0; i<matrix.keySet().size(); i++){
			line = matrix.get(matrix.keySet().toArray()[i]);
			for (int j=0; j<matrix.get(matrix.keySet().toArray()[i]).size(); j++)
				if (line.get(j) == element)
					return (String)matrix.keySet().toArray()[i];
		}
		return "";		
	}
	
//**********************************************************************************************
//****************************** calculaFMeasure ***********************************************
//**********************************************************************************************	
	private float[] calculateFMeasure(int numTruePositivePerClass[], int numFalsePositivePerClass[], int numFalseNegativePerClass[]){		
		float pi, ro;
		int i,numerator=0, denominatorPredict=0, denominatorReal=0;
		float FMeasurePorClasse[];
		float FMeasureMicro=0, FMeasureMacro=0;
		FMeasurePorClasse = new float[classNames.size()];

		int numClasses = 0;
		//calculate f-measure micro
		for (i=0; i<classNames.size();i++){
				numerator = numerator + numTruePositivePerClass[i];
				denominatorPredict = denominatorPredict + (numTruePositivePerClass[i] + numFalsePositivePerClass[i]);
				denominatorReal = denominatorReal + (numTruePositivePerClass[i] + numFalseNegativePerClass[i]);
		}
		pi = (float)numerator/denominatorPredict;
		ro = (float)numerator/denominatorReal;
		FMeasureMicro = (float)(2*pi*ro)/(pi+ro);
		//calculate FMeasureMacro
		for (i=0;i<classNames.size();i++){
			if (numTruePositivePerClass[i]+numFalseNegativePerClass[i] > 0){ //total de elementos de uma classe
				if (numTruePositivePerClass[i]+numFalsePositivePerClass[i] == 0)
					pi = 0.0f;
				else pi = (float)numTruePositivePerClass[i]/(numTruePositivePerClass[i]+numFalsePositivePerClass[i]);
				ro = (float)numTruePositivePerClass[i]/(numTruePositivePerClass[i]+numFalseNegativePerClass[i]);
				if ((pi+ro) == 0)
					FMeasurePorClasse[i] = 0.0f;
				else FMeasurePorClasse[i] = (float)(2*pi*ro)/(pi+ro);
				FMeasureMacro = FMeasureMacro + FMeasurePorClasse[i];
				numClasses++;
			}
		}
		FMeasureMacro = FMeasureMacro/numClasses;
		float results[] = new float[2];
		results[0] = FMeasureMicro;
		results[1] = FMeasureMacro;
		return results;
	}
	
//**********************************************************************************************
//********************* index ******************************************************************	
//**********************************************************************************************	
	private int index(ArrayList<String> list, String element){
		for (int i=0; i< list.size(); i++)
			if (list.get(i).compareTo(element)==0)
				return i;
		return -1;
	}
}
