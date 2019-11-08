package noveltydetection.HeCluS;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import java.awt.BasicStroke;
import java.awt.Color;
import java.io.*;
import java.util.List;

public class FreeChartGraph {
	   private JFreeChart graphic;
	   private String directory;

	   public FreeChartGraph(String directory){
		   this.directory = directory;
	   }
//***********************************************************************************
//***************************criarDatasetXY******************************************
//***********************************************************************************		   	   
	   private XYDataset criarDatasetXY(List<Float> measure1, List<Float> measure2, List<Float> measure3, List<Float> d, int EvaluationType){
		   final XYSeriesCollection xyDataset = new XYSeriesCollection();
		   final XYSeries series1; 
		   final XYSeries series2; 
		   final XYSeries series3; 
		   
		   if (EvaluationType == 0){
			   series1 = new XYSeries("Unk");
			   series2 = new XYSeries("FMacro");
			   series3 = new XYSeries("CER");
		   }
		   else{
			   series1 = new XYSeries("FNew");
			   series2 = new XYSeries("Mnew");
			   series3 = new XYSeries("Err");
		   }
		   
		   if(measure1.size () != 0){ 
	            for(int i=0;i<measure1.size ();i++) {
	            	series1.add(d.get(i),measure1.get(i));
	            }
		        xyDataset.addSeries(series1);
		   }
		   
		   if (measure2.size () != 0 ) {
	            for(int i=0;i<measure2.size ();i++) {
	            	series2.add(d.get(i),measure2.get(i));
	            }
	            xyDataset.addSeries(series2);
		   }
	       if (measure3.size()!=0){
	    	   for(int i=0;i<measure3.size ();i++) {	            	
	    		   series3.add(d.get(i),measure3.get(i));
		       }
	           xyDataset.addSeries(series3);            
	       }		 		   	
	       return xyDataset;		   
	   }
	   
//***********************************************************************************
//***************************addNoveltyExtension*************************************
//***********************************************************************************
	   public void addNoveltyExtension(List<Integer> values, Color color, float thickness ){
		   final XYPlot plot = graphic.getXYPlot();
		   ValueMarker marker;
		   for (int i=0; i< values.size();i++){
			   marker = new ValueMarker(values.get(i));
			   marker.setPaint(color);
			   marker.setStroke(new BasicStroke(thickness));
			   plot.addDomainMarker(marker);
		   }
	   }	
	   
//***********************************************************************************
//***************************createGraphic*******************************************
//***********************************************************************************
	    public void createGraphic( List<Float> measure1, List<Float> measure2, List<Float> measure3, List<Float> values_Timestamp, List<Integer> values_Nov, List<Integer> values_Ext, int EvaluationType ) throws IOException {
	    	
	        XYDataset dataXY = criarDatasetXY(measure1, measure2, measure3, values_Timestamp,EvaluationType);
	        //create graphic	        
	        graphic = ChartFactory.createXYLineChart("", "Timestamp", "Evaluation Measure Value", dataXY, PlotOrientation.VERTICAL, true, true, false);
	        
	        addNoveltyExtension(values_Nov, Color.gray,1.0f);
	        //addNoveltyExtension(values_Ext, Color.gray,1.0f);
	        	        
	        XYPlot xyplot = (XYPlot) graphic.getPlot();	        
	        ValueAxis rangeAxis = xyplot.getRangeAxis();
	        rangeAxis.setRange(0.0, 50);
	        
	        final NumberAxis rangeAxis2 = (NumberAxis) xyplot.getRangeAxis(); 
	        rangeAxis2.setTickUnit(new NumberTickUnit(10)); 

	        final NumberAxis domainAxis = (NumberAxis) xyplot.getDomainAxis(); 
	        domainAxis.setTickUnit(new NumberTickUnit(20));
	               
	        xyplot.setBackgroundPaint(Color.white);	        
	        xyplot.setDomainGridlinePaint(Color.white);
	        xyplot.setRangeGridlinePaint(Color.white);
	        
	        XYItemRenderer r = xyplot.getRenderer(); 
	        r.setSeriesStroke(0, new BasicStroke(3.0f));
	        r.setSeriesStroke(1, new BasicStroke(3.0f ));
	        r.setSeriesStroke(2, new BasicStroke(3.0f ));
	        	        
	        //print the graphic in a file
	        OutputStream arquivo = new FileOutputStream(directory+"_grafico.png");
	        saveGraphic(graphic,arquivo);
	        arquivo.close();
	     }

//***********************************************************************************
//***************************saveGraphic*******************************************
//***********************************************************************************	    
	    public void saveGraphic(JFreeChart graphic, OutputStream out) throws IOException {
		     ChartUtilities.writeChartAsPNG(out, graphic, 500, 500);
		}
	    
//***********************************************************************************
//***************************getGraphic**********************************************
//***********************************************************************************	    
	    public JFreeChart getGraphic(){
	    	return graphic;
	    }	    
}
