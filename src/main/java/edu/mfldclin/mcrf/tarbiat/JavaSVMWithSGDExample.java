/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.mfldclin.mcrf.tarbiat;

import edu.mfldclin.mcrf.tarbiat.utils.Resource;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

// $example on$
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

/**
 * Example for SVMWithSGD.
 */
public class JavaSVMWithSGDExample {

    public static void main(String[] args) {
        //SparkConf conf = new SparkConf().setAppName("JavaSVMWithSGDExample");
        SparkConf sparkConf = new SparkConf().setAppName("JavaSVMWithSGDExample");
        sparkConf.setMaster("local[2]");        

        //sparkConf.set(key, value)
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = Resource.getPath("data/mllib/sample_svm_data.txt");

        if (args.length > 0) {
            path = args[0];
        }/* else {
            System.out.println("No argument passed!");
            System.exit(-1);
        }*/

        System.out.println("---------- input file: " + path);
        
        long currentTimeMillis = System.currentTimeMillis();
        
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = data.sample(false, 0.75, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = data.subtract(training);

        // Run training algorithm to build the model.
        int numIterations = 100;
        final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Clear the default threshold.
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
            public Tuple2<Object, Object> call(LabeledPoint p) {
                Double score = model.predict(p.features());
                return new Tuple2<Object, Object>(score, p.label());
            }
        }
        );

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics
                = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();

        long currentTimeMillis1 = System.currentTimeMillis();
        long elapsedTime = currentTimeMillis1 - currentTimeMillis;

        String time = String.format("%d min, %d sec",
                TimeUnit.MILLISECONDS.toMinutes(elapsedTime),
                TimeUnit.MILLISECONDS.toSeconds(elapsedTime)
                        - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(elapsedTime))
        );

        System.out.println("Time: " + time);
        System.out.println("Area under ROC = " + auROC);

        // Save and load model
        String modelPath = "target/tmp/" + System.currentTimeMillis() + "/javaSVMWithSGDModel";
        model.save(jsc.sc(), modelPath);
        SVMModel sameModel = SVMModel.load(jsc.sc(), modelPath);
        // $example off$

        jsc.stop();
    }

}
