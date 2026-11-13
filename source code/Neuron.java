import java.util.Random;
import java.util.ArrayList;

public class Neuron
{
    //features
    private int numberOfWeightsFromNextLayer;
    private int neuronId;
    private double outcome;
    private double alpha;
    private double eta;
    private double gradient;
    private ArrayList <Synapse> weights = new ArrayList <Synapse> ( );
    
    //constructor

    public Neuron ( int numberOfWeightsFromNextLayer, int neuronId, double alpha, double eta )
    {
        this.numberOfWeightsFromNextLayer = numberOfWeightsFromNextLayer;
        this.neuronId = neuronId;
        this.alpha = alpha;
        this.eta = eta;
        this.gradient = 0.0;
        
        //initialize weights between 0 and 1
        for ( int wI = 0; wI < numberOfWeightsFromNextLayer; wI ++ )
        {
            weights.add ( new Synapse ( ) );
            
            weights.get ( wI ).setWeight ( new Random ( ).nextDouble ( ) );
        }
    }
    
    //methods
    
    public double getOutcome ( )
    {
        return outcome;
    }
    
    public double getGradient  ( )
    {
        return gradient;
    }
    
    public ArrayList <Synapse> getWeights ( )
    {
        return weights;
    }
    
    public double getActivation ( double value )
    {
        //tanh (x)
        return Math.tanh ( value );
    }    
    
    public double getPrimeActivation ( double value )
    {
        //1 - tanh (x^2)?
        return 1 - Math.pow ( Math.tanh ( value ), 2 );
    }
    
    public double getDistributedWeightSigma ( Layer nextLayer )
    {
        double sum = 0;
        
        for ( int nLI = 0 ; nLI < nextLayer.size ( ) - 1; nLI ++ )
            sum += getWeights ( ).get ( nLI ).getWeight ( ) * nextLayer.get ( nLI ).getGradient ( );
            
        return sum;
    }
    
    public void setGradient ( double value )
    {
        gradient = value;
    }    
    
    public void setOutcome ( double value )
    {
        outcome = value;
    }
    
    public void calculateHiddenGradients ( Layer nextLayer )
    {
        double delta = getDistributedWeightSigma ( nextLayer );
        setGradient ( getPrimeActivation ( getOutcome ( ) ) * delta  );
    }
     
    public void calculateOutcomeGradients ( int target )
    {
        double delta = target - getOutcome ( );
        setGradient ( getPrimeActivation ( getOutcome ( ) ) * delta  );
    }
 
    public void doForwardPropagation ( Layer priorLayer )
    {
        double sum = 0;
        
        for ( int pLI = 0 ; pLI < priorLayer.size ( ); pLI ++ )
            sum += priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) * priorLayer.get ( pLI ).getOutcome ( );
            
        setOutcome ( getActivation ( sum ) );
    }
    public void updateWeights ( Layer priorLayer )
    {
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
        {
            double priorDeltaWeight = priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getDeltaWeight ( );
            
            //( eta * grad * priorO ) + ( alpha * pDW )
            double newDeltaWeight = ( eta * gradient * priorLayer.get ( pLI ).getOutcome ( ) ) + ( alpha * priorDeltaWeight );
            
            
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setDeltaWeight ( newDeltaWeight );
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setWeight ( priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) + newDeltaWeight );
        }
    }

    public void updateWeights1 ( Layer priorLayer )
    {
        double newDeltaWeight = 0;
        
        for ( int pLI = 0 ; pLI < priorLayer.size ( ); pLI ++ )
        {
            double priorDeltaWeight = priorLayer.get ( neuronId ).getWeights ( ).get ( pLI ).getDeltaWeight ( );
            
            newDeltaWeight += ( eta * getGradient ( ) * priorLayer.get ( neuronId ).getOutcome ( ) ) + ( alpha * priorDeltaWeight );
            
            priorLayer.get ( neuronId ).getWeights ( ).get ( pLI ).setDeltaWeight ( newDeltaWeight );
            
            priorLayer.get ( neuronId ).getWeights ( ).get ( pLI ).setWeight ( priorLayer.get ( neuronId ).getWeights ( ).get ( pLI ).getWeight ( ) + newDeltaWeight );
        }
        
    }
}