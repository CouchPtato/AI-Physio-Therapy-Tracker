import React, { useState } from 'react';
import { Button, Image, Platform, StyleSheet, View, Text, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const BACKEND_URL = 'http://localhost:8000/analyze_frame'; // Change if backend runs elsewhere

export default function HomeScreen() {
  const [image, setImage] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission required', 'Camera roll permissions are required!');
      return;
    }
    const pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
      base64: false,
    });
    if (!pickerResult.canceled && pickerResult.assets.length > 0) {
      setImage(pickerResult.assets[0].uri);
      setResult(null);
    }
  };

  const uploadImage = async () => {
    if (!image) return;
    setUploading(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: image,
        name: 'photo.jpg',
        type: 'image/jpeg',
      } as any);
      formData.append('exercise', 'squat');
      formData.append('user_id', '1');

      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      Alert.alert('Error', 'Failed to upload image or connect to backend.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Physio Therapy Tracker</Text>
      <Button title="Pick an Image" onPress={pickImage} />
      {image && (
        <>
          <Image source={{ uri: image }} style={styles.image} />
          <Button title="Analyze Exercise" onPress={uploadImage} disabled={uploading} />
        </>
      )}
      {uploading && <ActivityIndicator size="large" />}
      {result && (
        <View style={styles.result}>
          <Text>Exercise: {result.exercise}</Text>
          <Text>Angle: {result.angle}</Text>
          <Text>Form: {result.form}</Text>
          <Text>Stage: {result.stage}</Text>
          <Text>Reps: {result.reps}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 16 },
  title: { fontSize: 22, fontWeight: 'bold', marginBottom: 16 },
  image: { width: 300, height: 300, marginVertical: 16, borderRadius: 8 },
  result: { marginTop: 24, alignItems: 'center' },
});