import React, { useState } from 'react';
import { Button, Image, StyleSheet, View, Text, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const BACKEND_URL = 'http://localhost:8000/analyze_frame'; // Update to your backend URL if needed

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
      formData.append('exercise', 'squat'); // Default exercise type
      formData.append('user_id', '1'); // Default user ID

      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });

      if (!res.ok) {
        const errorMessage = await res.text();
        console.error('Backend error:', errorMessage);
        Alert.alert('Error', 'Backend returned an error.');
        return;
      }

      const data = await res.json();
      console.log('Response data:', data); // Debugging
      setResult(data);
    } catch (e) {
      console.error('Error:', e); // Debugging
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
          <Text>Exercise: {result.exercise || 'N/A'}</Text>
          <Text>Angle: {result.angle || 'N/A'}</Text>
          <Text>Form: {result.form || 'N/A'}</Text>
          <Text>Stage: {result.stage || 'N/A'}</Text>
          <Text>Reps: {result.reps || 'N/A'}</Text>
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