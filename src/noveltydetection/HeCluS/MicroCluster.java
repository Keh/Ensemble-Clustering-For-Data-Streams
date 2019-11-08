package noveltydetection.HeCluS;

import weka.core.DenseInstance;
import weka.core.Instance;
import moa.clusterers.clustream.ClustreamKernel;

public class MicroCluster extends ClustreamKernelMOAModified{
	private static final long serialVersionUID = 1L;
	private String labelClass;	  // label of the class associated to the micro-cluster. If it is a novelty the class is a sequential number
    private String category;  	  // category: normal, novelty or extension
    private int time;		      // timestamp associated to the last example classified in this micro-cluster
    
    public MicroCluster(ClustreamKernelMOAModified element, String labelClasse, String category, int time){
    	super(element);
    	this.labelClass = labelClasse;
    	this.category = category;
    	this.time = time;
    }  
    
    public void insert(double element[], int timestamp){
    	Instance inst= new DenseInstance(1, element);	
    	super.insert(inst, timestamp);
    	this.setTime(timestamp);
    }
                
    public void setCategory(String category){
		this.category = category;
	}
            								
	public void setLabelClass(String labelClass){
			this.labelClass = labelClass;
	}
            
	public String getCategory(){
		return category;
	}
            							
	public String getLabelClass(){
		return labelClass;
	}

	public int getTime() {
		return time;
	}

	public void setTime(int time) {
		this.time = time;
	}	
			
}	
		
		


