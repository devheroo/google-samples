import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Scanner;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.auth.http.HttpCredentialsAdapter;
import java.io.FileInputStream;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Java Console Application for text summarization using Google Vertex AI Gemma3 model
 * 
 * Prerequisites:
 * 1. Add these dependencies to your pom.xml or build.gradle:
 *    - com.google.auth:google-auth-library-oauth2-http
 *    - com.google.code.gson:gson
 * 2. Set up Google Cloud credentials (service account key file)
 * 3. Configure your project ID, location, and endpoint ID
 */
public class VertexAISummarizer {
    
    // Configuration - Replace with your actual values
    private static final String PROJECT_ID = "your-project-id";
    private static final String LOCATION = "your-location"; // e.g., "us-central1"
    private static final String ENDPOINT_ID = "your-endpoint-id";
    private static final String CREDENTIALS_PATH = "path/to/your/service-account-key.json";
    
    private static final String ENDPOINT_URL = String.format(
        "https://%s.%s-%s.prediction.vertexai.goog/v1/projects/%s/locations/%s/endpoints/%s:predict",
        ENDPOINT_ID, LOCATION, PROJECT_ID, PROJECT_ID, LOCATION, ENDPOINT_ID
    );
    
    private final HttpClient httpClient;
    private final Gson gson;
    private GoogleCredentials credentials;
    
    public VertexAISummarizer() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
        this.gson = new Gson();
        
        try {
            // Initialize Google Cloud credentials
            this.credentials = GoogleCredentials.fromStream(new FileInputStream(CREDENTIALS_PATH))
                .createScoped("https://www.googleapis.com/auth/cloud-platform");
        } catch (IOException e) {
            System.err.println("Failed to load credentials: " + e.getMessage());
            System.exit(1);
        }
    }
    
    /**
     * Creates the request payload for Gemma3 model
     */
    private String createRequestPayload(String textToSummarize) {
        JsonObject instance = new JsonObject();
        instance.addProperty("prompt", 
            "Please provide a concise summary of the following text:\n\n" + textToSummarize + "\n\nSummary:");
        
        JsonObject parameters = new JsonObject();
        parameters.addProperty("maxOutputTokens", 150);
        parameters.addProperty("temperature", 0.3);
        parameters.addProperty("topP", 0.8);
        parameters.addProperty("topK", 40);
        
        JsonObject payload = new JsonObject();
        payload.add("instances", gson.toJsonTree(new JsonObject[]{instance}));
        payload.add("parameters", parameters);
        
        return gson.toJson(payload);
    }
    
    /**
     * Sends request to Vertex AI endpoint and returns the summary
     */
    public String summarizeText(String text) throws IOException, InterruptedException {
        // Refresh credentials if needed
        credentials.refreshIfExpired();
        String accessToken = credentials.getAccessToken().getTokenValue();
        
        String requestBody = createRequestPayload(text);
        
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(ENDPOINT_URL))
            .header("Authorization", "Bearer " + accessToken)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .timeout(Duration.ofSeconds(60))
            .build();
        
        HttpResponse<String> response = httpClient.send(request, 
            HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new RuntimeException("Request failed with status: " + response.statusCode() + 
                "\nResponse: " + response.body());
        }
        
        return extractSummaryFromResponse(response.body());
    }
    
    /**
     * Extracts the summary text from the API response
     */
    private String extractSummaryFromResponse(String responseBody) {
        try {
            JsonObject jsonResponse = JsonParser.parseString(responseBody).getAsJsonObject();
            
            if (jsonResponse.has("predictions") && jsonResponse.get("predictions").isJsonArray()) {
                JsonObject prediction = jsonResponse.getAsJsonArray("predictions")
                    .get(0).getAsJsonObject();
                
                if (prediction.has("content")) {
                    return prediction.get("content").getAsString().trim();
                } else if (prediction.has("generated_text")) {
                    return prediction.get("generated_text").getAsString().trim();
                }
            }
            
            // Fallback: return the entire response for debugging
            return "Could not parse summary. Full response: " + responseBody;
            
        } catch (Exception e) {
            return "Error parsing response: " + e.getMessage() + "\nFull response: " + responseBody;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("=== Vertex AI Gemma3 Text Summarizer ===\n");
        
        // Validate configuration
        if (PROJECT_ID.equals("your-project-id") || 
            LOCATION.equals("your-location") || 
            ENDPOINT_ID.equals("your-endpoint-id")) {
            System.err.println("Please configure PROJECT_ID, LOCATION, and ENDPOINT_ID in the code.");
            System.exit(1);
        }
        
        VertexAISummarizer summarizer = new VertexAISummarizer();
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            System.out.println("\nOptions:");
            System.out.println("1. Summarize text");
            System.out.println("2. Exit");
            System.out.print("Choose an option (1-2): ");
            
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    System.out.println("\nEnter the text you want to summarize:");
                    System.out.println("(Type 'END' on a new line when finished)\n");
                    
                    StringBuilder textBuilder = new StringBuilder();
                    String line;
                    while (!(line = scanner.nextLine()).equals("END")) {
                        textBuilder.append(line).append("\n");
                    }
                    
                    String textToSummarize = textBuilder.toString().trim();
                    if (textToSummarize.isEmpty()) {
                        System.out.println("No text provided. Please try again.");
                        break;
                    }
                    
                    System.out.println("\nGenerating summary...");
                    
                    try {
                        String summary = summarizer.summarizeText(textToSummarize);
                        System.out.println("\n" + "=".repeat(50));
                        System.out.println("SUMMARY:");
                        System.out.println("=".repeat(50));
                        System.out.println(summary);
                        System.out.println("=".repeat(50));
                        
                    } catch (Exception e) {
                        System.err.println("Error generating summary: " + e.getMessage());
                        e.printStackTrace();
                    }
                    break;
                    
                case "2":
                    System.out.println("Goodbye!");
                    scanner.close();
                    System.exit(0);
                    break;
                    
                default:
                    System.out.println("Invalid option. Please choose 1 or 2.");
            }
        }
    }
}