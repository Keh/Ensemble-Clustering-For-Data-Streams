package noveltydetection.HeCluS;
import java.io.*;
import java.util.ArrayList;

import moa.cluster.Clustering;
import weka.core.DenseInstance;
import weka.core.Instance;


public class ClustreamOffline {   	
   	private ArrayList<Integer> clusteringResul[];    	
    private int clusterExamples[]; 
	
//*******************************************************************************   
//******************** Clustering ***********************************************
//*******************************************************************************    
	public Clustering CluStream(String lblClass, String filename, float numMicro,boolean flagMicro,boolean flagKMeans) throws IOException{		
		BufferedReader fileIn;	        
		Clustering microClusters = null;
        String line;
  		   		
   		ClustreamMOAModified algClustream = new ClustreamMOAModified();
    	algClustream.prepareForUse();
    	
    	if (flagKMeans == true)   	
    		algClustream.initializeBufferSize(10*(int)numMicro,(int)numMicro);
    	else
    		algClustream.initializeBufferSize(1*(int)numMicro,(int)numMicro);
    		
    	//read the data set and executing clustream algorithm
    	int lineSize=0;
        fileIn = new BufferedReader(new FileReader(new File(filename+lblClass)));
        double data[]=null;
        int cont=0;
   		while ((line = fileIn.readLine()) != null){			
   			String linhaarqString []= line.split(",");
   			lineSize = linhaarqString.length-1;
   			data = new double[lineSize];
   			cont++;
   			for (int i=0; i<lineSize; i++){
   				data[i] = Double.parseDouble(linhaarqString[i]);
   	        } 	        
   			Instance inst= new DenseInstance(1, data);	
   			algClustream.trainOnInstanceImpl(inst);			
   		}
   		fileIn.close();
   		
   		//obtain the micro-clusters  
       	microClusters = algClustream.getMicroClusteringResult();
       	ArrayList<ArrayList<Integer>> grupos = new ArrayList<ArrayList<Integer>>();        	       	       	
       	ArrayList<double[]> centers = new ArrayList<double[]>();
       	
       	if (flagMicro == false){ //using clustream + KMeans       	
       		//choose the best value for K and executing KMeans
       		Clustering macroClustering = null;
       		macroClustering = KMeansMOAModified.OMRk(microClusters,0,grupos);        	
       		int numClusters = macroClustering.size();      
       		
       		ArrayList<ArrayList<Integer>> grupoMicro = new ArrayList<ArrayList<Integer>>();
           	grupoMicro = ((ClustreamMOAModified)algClustream).getClusterExamples();
           	clusteringResul = (ArrayList<Integer>[])new ArrayList[numClusters];
           	
           	for (int i=0; i<clusteringResul.length;i++){
           		clusteringResul[i] = new ArrayList<Integer>();
    		}
           	
    		for (int i=0; i<grupos.size();i++){
    			for (int j=0; j<grupos.get(i).size();j++){
    				int pos = grupos.get(i).get(j);
    				for (int k=0; k< grupoMicro.get(pos).size();k++){
    					clusteringResul[i].add((int)grupoMicro.get(pos).get(k));
    				}
    			}
    		}
       		for (int k= 0; k<numClusters; k++){
    			centers.add(macroClustering.getClustering().get(k).getCenter());
    		}

       		
       	}
       	else{ // using only clustream
       		if (microClusters.size() >= 1){
	           	grupos = ((ClustreamMOAModified)algClustream).getClusterExamples();  
	           	clusterExamples = new int[cont];
	    		int value;
	    		for (int i=0; i<grupos.size();i++){
	    			//remove micro-cluster with less than 3 examples
	    			if (((ClustreamKernelMOAModified)microClusters.get(i)).getWeight() < 3)
	    				value = -1;
	    			else 
	    				value = i;
	    			
	    			for (int j=0;j<grupos.get(i).size();j++){
	    				clusterExamples[grupos.get(i).get(j)] = value; //i+1;
	    			}
	    			if (((ClustreamKernelMOAModified)microClusters.get(i)).getWeight() < 3){
	    				microClusters.remove(i);
	    				grupos.remove(i);
	    				i--;
	    			}
	    		}
//	    		clusterExamples = new int[cont];
	    		
//	    		for(int i=0; i<1;i++) {
//	    			System.out.print(" teste grupos " + grupos.get(i));
//	    		}
//	    		System.out.println();
//	    		for(int i=0; i<grupos.get(0).size();i++) {
//	    			System.out.print(", " + clusterExamples[i]);
//	    		}
//	    		System.out.println();
       		}
    	}    
       	System.out.println("micro cluster size " + microClusters.size());
       	return microClusters;       	       	       	
	}
	
//*******************************************************************************   
//******************** getClusterExamples ***************************************
//******************************************************************************* 	
	public int [] getClusterExamples(){
		return clusterExamples;
	}
	
//*******************************************************************************   
//******************** Clustering ***********************************************
//******************************************************************************* 	
	public int[] getClusteringResults(){
		int resultado[];
		int nroelem = 0;
		for (int i=0; i<clusteringResul.length;i++){
			nroelem += clusteringResul[i].size();
		}
		resultado = new int[nroelem];
		
		for (int i=0; i<clusteringResul.length;i++){
			for (int j=0;j<clusteringResul[i].size();j++){
				resultado[clusteringResul[i].get(j)] = i+1;
			}
		}
		return resultado;
	}
}

