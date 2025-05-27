import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class ProductRecommendation {

    public static void main(String[] args) {
        try {
            // Load data from CSV file (userID, itemID, preference)
            File dataFile = new File("data.csv");
            DataModel model = new FileDataModel(dataFile);

            // Use Pearson Correlation similarity between users
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // Define neighborhood of nearest 3 users
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, similarity, model);

            // Build recommender
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // For each user, recommend 3 products
            LongPrimitiveIterator userIterator = model.getUserIDs();

            while (userIterator.hasNext()) {
                long userId = userIterator.nextLong();
                List<RecommendedItem> recommendations = recommender.recommend(userId, 3);

                System.out.println("Recommendations for User ID: " + userId);
                if (recommendations.isEmpty()) {
                    System.out.println("  No recommendations available.");
                } else {
                    for (RecommendedItem recommendation : recommendations) {
                        System.out.println("  Item ID: " + recommendation.getItemID() + ", Predicted Preference: " + recommendation.getValue());
                    }
                }
                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}