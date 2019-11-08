package noveltydetection.HeCluS;

import javax.swing.*;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;

public class GraphicFrame extends JFrame{
	private static final long serialVersionUID = 1L;

	public GraphicFrame(JFreeChart grafico){
		super("Gr�fico - Medida de Avalia��o");
		ChartPanel chartPanel = new ChartPanel(grafico);
        setContentPane(chartPanel);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setSize(1000,400);
		setVisible(true);
	}
}
