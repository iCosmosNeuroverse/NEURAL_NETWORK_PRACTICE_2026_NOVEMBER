public class NeuralNetwork
{
    
    //features
    private double alpha = 0.5;
    private double eta = 0.2;
    private Architecture architecture = new Architecture ( "2,2,1" );
    private Layers layers = new Layers ( );
    
    //constructor
    
    public NeuralNetwork ( )
    {
        for ( int lSI = 0; lSI < architecture.size ( ); lSI ++ )
        {
            layers.add ( new Layer ( ) );
            
            for ( int lI = 0; lI <= architecture.get ( lSI ); lI ++ )
            {
                int numberOfNeuronsFromNextLayer = lSI + 1 < architecture.size ( ) ? architecture.get ( lSI + 1 ) : 0;
            
                Neuron neuron = new Neuron ( numberOfNeuronsFromNextLayer, lI, alpha, eta ); 
                
                layers.get ( lSI ).add ( neuron );
                
                //initialize bias neuron
                layers.get ( lSI ).get ( layers.get ( lSI ).size ( ) - 1 ).setOutcome ( 1.0 );
            }
        }
    }

    
    //methods
    
    public void doForwardPropagation ( int [] inputs )
    {
        //initialize first layer of neural network with problem inputs
        for ( int iI = 0; iI < inputs.length; iI ++ )
            layers.get ( 0 ).get ( iI ).setOutcome ( inputs [ iI ] );
            
            
        //for other layers do forward propagation
        for ( int lSI = 1; lSI < architecture.size ( ); lSI ++ )
        {
            Layer priorLayer = layers.get ( lSI - 1 );
            
            for ( int lI = 0; lI < architecture.get ( lSI ); lI ++ )
            {
                layers.get ( lSI ).get ( lI ).doForwardPropagation ( priorLayer );
            }   
        }
    }
    
    public void doBackwardPropagation ( int target )
    {
        //calc out grads
        layers.get ( layers.size ( ) - 1 ).get ( 0 ).calculateOutcomeGradients ( target );
        
        //calc hid grads
        for ( int lSI = architecture.size ( ) - 2; lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer nextLayer = layers.get ( lSI + 1 );
            
            for ( int lI = 0; lI < currentLayer.size ( ); lI ++ )
                currentLayer.get ( lI ).calculateHiddenGradients ( nextLayer );
        }
        
        //update weights
        for ( int lSI = architecture.size ( ) - 1; lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer priorlayer = layers.get ( lSI - 1 );
            
            for ( int lI = 0; lI < currentLayer.size ( ) - 1; lI ++ )
                currentLayer.get ( lI ).updateWeights ( priorlayer );
        }
    }
    
    public double getOutcome ( )
    {
        return layers.get ( layers.size ( ) - 1 ).get ( 0 ).getOutcome ( );
    }
}