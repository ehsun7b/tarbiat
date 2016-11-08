/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.mfldclin.mcrf.tarbiat.utils;

/**
 *
 * @author Ehsun Behravesh <post@ehsunbehravesh.com>
 */
public class Resource {

    public static String getPath(String resource) {
        return Thread.currentThread().getContextClassLoader().getResource(resource).getFile();
    }
    
}
