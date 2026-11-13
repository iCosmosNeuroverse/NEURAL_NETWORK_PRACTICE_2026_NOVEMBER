import java.util.ArrayList;


public class Architecture extends ArrayList <Integer>
{

    //features
    private String architecture = "";
    
    //constructor
    public Architecture ( String architecture )
    {
        String [ ] parts = architecture.split ( "," );
        
        for ( int pI = 0; pI < parts.length; pI ++ )
            add ( Integer.parseInt ( parts [ pI ] ) );
        
    }
    
}
