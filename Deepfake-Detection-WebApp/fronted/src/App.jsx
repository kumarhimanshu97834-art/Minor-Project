
import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [fileType, setFileType] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  function handleFile(event) {
    const selectedFile = event.target.files[0]
    if (!selectedFile) return

    const isVideo = selectedFile.type.startsWith('video/')
    const isImage = selectedFile.type.startsWith('image/')

    if (!isVideo && !isImage) {
      setError('Please select an image or video.')
      return
    }

    setFile(selectedFile)
    setFileType(isVideo ? 'video' : 'image')
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError('')
  }

  async function handleAnalyze() {
    if (!file) return

    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    const endpoint = fileType === 'video'
      ? '/api/predict'
      : '/api/predict-image'

    try {
      const response = await fetch(
        `http://127.0.0.1:8001${endpoint}`,
        {
          method: 'POST',
          body: formData
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed')
      }

      setResult(data)

    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <nav className="navbar">
        <h2>Deepfake<span>AI</span></h2>
        <div>Home&nbsp;&nbsp; About&nbsp;&nbsp; Detection</div>
      </nav>

      <main className="hero">
        <p className="tag">AI POWERED MEDIA ANALYSIS</p>

        <h1>
          Detect Deepfakes.<br />
          <span>Trust What You See.</span>
        </h1>

        <p className="description">
          Upload an image or video and let our AI model
          analyze it to identify possible manipulation.
        </p>

        <div className="upload-box">
          <h3>Upload Image or Video</h3>
          <p>JPG, PNG, WEBP, MP4, MOV, AVI supported</p>

          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/x-msvideo"
            onChange={handleFile}
          />

          {preview && fileType === 'video' && (
            <video
              src={preview}
              controls
              width="100%"
              style={{ marginTop: '20px', borderRadius: '12px' }}
            />
          )}

          {preview && fileType === 'image' && (
            <img
              src={preview}
              alt="Selected preview"
              style={{
                marginTop: '20px',
                maxWidth: '100%',
                maxHeight: '400px',
                borderRadius: '12px'
              }}
            />
          )}

          {file && (
            <button onClick={handleAnalyze} disabled={loading}>
              {loading
                ? 'Analyzing...'
                : `Analyze ${fileType === 'video' ? 'Video' : 'Image'}`}
            </button>
          )}

          {error && (
            <p style={{ color: 'red' }}>{error}</p>
          )}

          {result && (
            <div className={`result-card ${result.prediction.toLowerCase()}`}>
              <p className="result-label">AI ANALYSIS RESULT</p>

              <h2>
                {result.prediction === 'Real'
                  ? '✓ REAL MEDIA'
                  : '⚠ FAKE MEDIA'}
              </h2>

              <div className="probability-section">
                <div className="probability-heading">
                  <span>Real Score</span>
                  <strong>{result.real_probability}%</strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-fill real-fill"
                    style={{ width: `${result.real_probability}%` }}
                  />
                </div>

                <div className="probability-heading">
                  <span>Fake Score</span>
                  <strong>{result.fake_probability}%</strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-fill fake-fill"
                    style={{ width: `${result.fake_probability}%` }}
                  />
                </div>
              </div>

              <div className="frames-info">
                {fileType === 'video'
                  ? <>Face images analyzed: <strong>{result.frames_analyzed}</strong></>
                  : <>Faces analyzed: <strong>{result.faces_analyzed}</strong></>}
              </div>

              <p className="disclaimer">
                Experimental model result — not a verified verdict.
              </p>
            </div>
          )}
        </div>

        <p className="footer">
          Powered by PyTorch | ResNet50
        </p>
      </main>
    </div>
  )
}

export default App
