# Astraeus Quick Reference Guide

This guide provides quick instructions for common operations in the Astraeus project.

## Starting the System

```bash
# Activate the virtual environment
source astraeus_env/bin/activate  # On Windows: astraeus_env\Scripts\activate

# Start the server
python run.py
```

## Document Operations

### Upload Documents

**Web Interface**:
1. Drag and drop files to the upload area, or
2. Click the upload area to select files, or
3. Use the document panel's upload button

**Supported File Types**:
- Text: `.txt`, `.md`
- Documents: `.pdf`, `.doc`, `.docx`
- Spreadsheets: `.csv`, `.xls`, `.xlsx`
- Data: `.json`, `.xml`, `.yaml`
- Images: `.jpg`, `.jpeg`, `.png`

### Manage Documents

**Open Document Panel**:
1. Click "Manage Documents" button in the header, or
2. Click "Manage All Documents" in the documents section

**Within the Document Panel**:
- View summaries: Click "Show Summary" button
- Ask questions: Click "Ask Question" button
- Delete: Click "Delete" button

### Clear All Documents

```bash
# Run with confirmation prompt
./reset.sh

# Or run directly without confirmation
python reset_database.py
```

## Query Operations

### Basic Query

1. Type your question in the chat input
2. Press Enter or click Send

### Document-Specific Queries

1. Open the document panel
2. Find the document you want to ask about
3. Click "Ask Question" button
4. Edit the prefilled query if needed
5. Press Enter

### View Source Documents

1. After receiving an answer
2. Click "Show Sources" to see the context used
3. Click "Hide Sources" to collapse them

## System Maintenance

### Backup Data

```bash
# Backup the data directories
cp -r data/uploads /your/backup/path/uploads
cp -r data/vector_store /your/backup/path/vector_store
```

### Restore Data

```bash
# Restore from backup
cp -r /your/backup/path/uploads data/uploads
cp -r /your/backup/path/vector_store data/vector_store
```

### Update API Keys

1. Edit the `.env` file
2. Add or modify `GOOGLE_API_KEY=your_key_here`
3. Restart the application

## Troubleshooting

### Reset Application

If the application behaves unexpectedly:

1. Stop the server (Ctrl+C)
2. Restart: `python run.py`

### API Key Issues

If you see errors about missing or invalid API keys:

1. Check `.env` file exists in project root
2. Verify `GOOGLE_API_KEY` is set correctly
3. Ensure the key has appropriate permissions

### Upload Failures

If document uploads fail:
1. Check file size (max 16MB)
2. Verify file type is supported
3. Check `data/uploads` directory permissions

### Search Problems

If searches return unexpected results:
1. Try more specific queries
2. Verify documents were processed successfully
3. Check the backend logs for embedding errors

## Common Commands

```bash
# Start the application
python run.py

# Reset the database
./reset.sh

# View logs (if running with output redirection)
cat astraeus.log

# Check application status
curl http://localhost:5000/

# List all uploaded documents
ls -la data/uploads/
``` 